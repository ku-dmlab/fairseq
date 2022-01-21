
import numpy as np
import collections
import math
import json
from argparse import Namespace
from dataclasses import dataclass, field

import torch
import torch.optim as optim
import torch.nn.functional as F
import sacrebleu
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from fairseq.data.data_utils import collate_tokens

@dataclass
class ActorCriticCriterionConfig(FairseqDataclass):
    sample_beam: int = field(default=5, metadata={"help": "number of sample size"})
    use_beam_while_training: bool = field(default=False)
    use_value_for_return: bool = field(default=False)
    use_value_for_baseline: bool = field(default=False)
    use_clone_loss: bool = field(default=True)
    alpha: float = field(default=1.0)
    tau: float = field(default=0.9)
    detach_actor: bool = field(default=False)
    loss_scaler: float = field(default=50.)

@register_criterion(
    "actor_critic", dataclass=ActorCriticCriterionConfig
)
class ActorCriticCriterion(FairseqCriterion):
    def __init__(self, task, sample_beam, use_beam_while_training,
                 use_value_for_return, use_value_for_baseline, use_clone_loss, alpha, tau, detach_actor, loss_scaler):
        super().__init__(task)
        self.sample_beam = sample_beam
        self.use_beam_while_training = use_beam_while_training
        self.use_value_for_return = use_value_for_return
        self.use_value_for_baseline = use_value_for_baseline
        self.use_clone_loss = use_clone_loss
        self.alpha = alpha
        self.tau = tau
        self.detach_actor = detach_actor
        self.loss_scaler = loss_scaler
        self.generator = None
        self.vf = None
        self.vf_optimizer = None

    def _decode(self, toks, escape_unk=False):
        s = self.task.tgt_dict.string(
            toks.int().cpu(),
            self.task.cfg.eval_bleu_remove_bpe,
            unk_string=("UNKNOWNTOKENINREF" if escape_unk else "UNKNOWNTOKENINHYP"),
        )
        if self.task.tokenizer:
            s = self.task.tokenizer.decode(s)
        return s
 
    def forward(self, model, sample, reduce=True, critic_only=False):
        if self.generator is None:
            gen_args = Namespace(**json.loads(self.task.cfg.eval_bleu_args))
            gen_args.sample_beam = self.sample_beam
            if not self.use_beam_while_training:
                gen_args.sampling = True
                gen_args.sampling_topp = 0.5
            self.generator = self.task.build_generator([model], gen_args)

        was_train = model.training
        model.eval()
        with torch.no_grad():
            hypos = self.generator.generate([model], sample)
        if was_train:
            model.train()

        rewards = []
        pad_idx = self.task.tgt_dict.pad()
        eos_idx = self.task.tgt_dict.eos()
        
        num_hypos = len(hypos)
        num_samples = len(hypos[0])
        hypos = [[preds["tokens"] for preds in each] for each in hypos]
        for hypo, rtarget in zip(hypos, sample["target"]):
            rewards.append([])
            ref = self._decode(
                    utils.strip_pad(rtarget, pad_idx),
                    escape_unk=True,  # don't count <unk> as matches to the hypo
                )
            for preds in hypo:
                hyp = self._decode(preds)
                if self.task.cfg.eval_tokenized_bleu:
                    rewards[-1].append(sacrebleu.corpus_bleu([hyp], [[ref]], tokenize="none").score)
                else:
                    rewards[-1].append(sacrebleu.corpus_bleu([hyp], [[ref]]).score)
        
        hypos = [item for sublist in hypos for item in sublist]
        vinputs = {"src_tokens": sample["net_input"]["src_tokens"].tile(
            1, num_samples).view(num_hypos * num_samples, -1),
            "src_lengths": sample["net_input"]["src_lengths"][:, None].tile(
                1, num_samples).view(num_hypos * num_samples)}
        vtargets = collate_tokens(hypos, pad_idx, eos_idx,
            left_pad=self.task.cfg.left_pad_target)
        vinputs["prev_output_tokens"] = collate_tokens(
            hypos, pad_idx, eos_idx, left_pad=self.task.cfg.left_pad_target,
            move_eos_to_beginning=True)

        orig_output = model(**sample["net_input"])
        orig_lprobs = model.get_normalized_probs(orig_output, log_probs=True)
        orig_lprobs = orig_lprobs.view(-1, orig_lprobs.size(-1))
        target = model.get_targets(sample, orig_output).view(-1)
        avg_clone_loss = F.nll_loss(orig_lprobs, target, ignore_index=pad_idx)

        out_features = model(**vinputs, features_only=True)[0]
        net_output = [model.output_layer(out_features)]
        value = model.vf(out_features.detach()) if self.detach_actor else model.vf(out_features)
        cql_loss1 = torch.logsumexp(value, dim=2, keepdim=True).view(num_hypos, num_samples, -1)
        value = torch.gather(value, 2, vtargets[:, :, None]).view(num_hypos, num_samples, -1)
        
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        lprob = lprobs.gather(dim=-1, index=vtargets[:, :, None])
        score = lprob.view(num_hypos, num_samples, -1)
        non_pad_mask = vtargets.ne(pad_idx).view(num_hypos, num_samples, -1)
        rewards = lprob.new_tensor(rewards).view(num_hypos, num_samples, 1)
        
        residual = rewards - value
        actor_loss = - (score * residual.detach())[non_pad_mask]
        critic_loss = 0.5 * residual.pow(2) * torch.abs(self.tau - torch.less(residual, 0).float())
        critic_loss = self.alpha * (cql_loss1 - value)[non_pad_mask] / self.loss_scaler

        batch_tokens = critic_loss.size(0) / num_samples
        avg_actor_loss = torch.sum(actor_loss) / batch_tokens
        avg_critic_loss = torch.sum(critic_loss) / batch_tokens

        avg_policy_loss = avg_clone_loss if self.use_clone_loss else avg_actor_loss
        avg_rl_loss = avg_critic_loss if critic_only else avg_policy_loss + avg_critic_loss

        logging_output = {
            'loss': utils.item(avg_rl_loss.data),
            'clone_loss': utils.item(avg_clone_loss.data),
            'actor_loss': utils.item(avg_actor_loss.data),
            'critic_loss': utils.item(avg_critic_loss.data),
            'sample_bleu': utils.item(torch.mean(rewards).data),
            'ntokens': batch_tokens,
        }
        return avg_rl_loss, batch_tokens, logging_output

    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        loss = sum(log.get("loss", 0) for log in logging_outputs)
        clone_loss = sum(log.get("clone_loss", 0) for log in logging_outputs)
        actor_loss = sum(log.get("actor_loss", 0) for log in logging_outputs)
        critic_loss = sum(log.get("critic_loss", 0) for log in logging_outputs)
        sample_bleu = sum(log.get("sample_bleu", 0) for log in logging_outputs)
        metrics.log_scalar("loss", loss, ntokens)
        metrics.log_scalar("clone_loss", clone_loss, ntokens)
        metrics.log_scalar("actor_loss", actor_loss, ntokens)
        metrics.log_scalar("critic_loss", critic_loss, ntokens)
        metrics.log_scalar("sample_bleu", sample_bleu, ntokens)
        
    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
