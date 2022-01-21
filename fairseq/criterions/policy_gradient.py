
import numpy as np
import collections
import math
import json
from argparse import Namespace
from dataclasses import dataclass, field

import torch
import sacrebleu
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from fairseq.data.data_utils import collate_tokens


@dataclass
class PolicyGradientCriterionConfig(FairseqDataclass):
    sample_beam: int = field(default=5, metadata={"help": "number of sample size"})
    use_sample_based_baseline: bool = field(default=False)
    use_beam_while_training: bool = field(default=False)


@register_criterion(
    "policy_gradient", dataclass=PolicyGradientCriterionConfig
)
class PolicyGradientCriterion(FairseqCriterion):
    def __init__(self, task, sample_beam, use_sample_based_baseline, use_beam_while_training):
        super().__init__(task)
        self.sample_beam = sample_beam
        self.use_sample_based_baseline = use_sample_based_baseline
        self.use_beam_while_training = use_beam_while_training
        self.generator = None

    def _decode(self, toks, escape_unk=False):
        s = self.task.tgt_dict.string(
            toks.int().cpu(),
            self.task.cfg.eval_bleu_remove_bpe,
            unk_string=("UNKNOWNTOKENINREF" if escape_unk else "UNKNOWNTOKENINHYP"),
        )
        if self.task.tokenizer:
            s = self.task.tokenizer.decode(s)
        return s
 
    def forward(self, model, sample, reduce=True):
        if self.generator is None:
            gen_args = Namespace(**json.loads(self.task.cfg.eval_bleu_args))
            gen_args.sample_beam = self.sample_beam
            if not self.use_beam_while_training:
                gen_args.sampling = True
                gen_args.sampling_topp = 0.5
            self.generator = self.task.build_generator([model], gen_args)

        model.eval()
        with torch.no_grad():
            hypos = self.generator.generate([model], sample)
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

        net_output = model(**vinputs)
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        lprob = -lprobs.gather(dim=-1, index=vtargets[:, :, None])
        non_pad_mask = vtargets.ne(pad_idx).view(num_hypos, num_samples, -1)
        rewards = lprob.new_tensor(rewards).view(num_hypos, num_samples, 1)
        if self.use_sample_based_baseline:
            adv = rewards - rewards.mean(1, keepdim=True)
            loss = (lprob.view(num_hypos, num_samples, -1) * adv)[non_pad_mask]
        else:
            loss = (lprob.view(num_hypos, num_samples, -1) * rewards)[non_pad_mask]
        batch_tokens = loss.size(0) / num_samples
        avg_rl_loss = torch.sum(loss) / batch_tokens

        logging_output = {
            'loss': utils.item(avg_rl_loss.data),
            'sample_bleu': utils.item(torch.mean(rewards).data),
            'ntokens': batch_tokens,
        }
        return avg_rl_loss, batch_tokens, logging_output

    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        loss = sum(log.get("loss", 0) for log in logging_outputs)
        sample_bleu = sum(log.get("sample_bleu", 0) for log in logging_outputs)
        metrics.log_scalar("loss", loss, ntokens)
        metrics.log_scalar("sample_bleu", sample_bleu, ntokens)
        
    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
