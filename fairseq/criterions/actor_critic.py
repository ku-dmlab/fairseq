
from email.headerregistry import UniqueDateHeader
import json
from argparse import Namespace
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
import sacrebleu
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from fairseq.data.data_utils import collate_tokens
from fairseq.tasks.translation_with_actor_critic_offline import TranslationWithActorCriticOffline

@dataclass
class ActorCriticCriterionConfig(FairseqDataclass):
    sample_beam: int = field(default=5, metadata={"help": "number of sample size"})
    use_beam_while_training: bool = field(default=False)
    use_reinforce: bool = field(default=False)
    use_ac: bool = field(default=False)
    alpha: float = field(default=1.0)
    tau: float = field(default=0.9)
    detach_actor: bool = field(default=False)
    reward_scaler: float = field(default=100.)
    learn_imitate: bool = field(default=False)
    target_learning_rate: float = field(default=0.001)
    use_pcl: bool = field(default=False)
    use_iql: bool = field(default=False)
    use_full: bool = field(default=False)
    use_bc: bool = field(default=False)
    use_gold_reward: bool = field(default=False)
    use_reward_weighted: bool = field(default=False)

@register_criterion(
    "actor_critic", dataclass=ActorCriticCriterionConfig
)
class ActorCriticCriterion(FairseqCriterion):
    def __init__(self, task, cfg):
        super().__init__(task)
        self.sample_beam = cfg.sample_beam
        self.use_beam_while_training = cfg.use_beam_while_training
        self.use_reinforce = cfg.use_reinforce
        self.use_ac = cfg.use_ac
        self.alpha = cfg.alpha
        self.tau = cfg.tau
        self.detach_actor = cfg.detach_actor
        self.reward_scaler = cfg.reward_scaler
        self.generator = None
        self.learn_imitate = cfg.learn_imitate
        if self.learn_imitate:
            self.tau = 0.5 # fix tau to be 0.5 in case of imitaion
        self.target_learning_rate = cfg.target_learning_rate
        self.use_pcl = cfg.use_pcl
        self.use_iql = cfg.use_iql
        self.use_full = cfg.use_full
        self.use_bc = cfg.use_bc
        self.use_gold_reward = cfg.use_gold_reward
        self.use_reward_weighted = cfg.use_reward_weighted

    @property
    def pad_idx(self):
        return self.task.tgt_dict.pad()

    @property
    def eos_idx(self):
        return self.task.tgt_dict.eos()

    def _decode(self, toks, is_hyp=True):
        if not is_hyp:
            toks = utils.strip_pad(toks, self.pad_idx)
        s = self.task.tgt_dict.string(
            toks.int().cpu(),
            self.task.cfg.eval_bleu_remove_bpe,
            unk_string=("UNKNOWNTOKENINHYP" if is_hyp else "UNKNOWNTOKENINREF"),
        )
        if self.task.tokenizer:
            s = self.task.tokenizer.decode(s)
        return s
 
    def _build_generator(self, model):
        assert self.generator is None
        gen_args = Namespace(**json.loads(self.task.cfg.eval_bleu_args))
        gen_args.sample_beam = self.sample_beam
        if not self.use_beam_while_training:
            gen_args.sampling = True
        self.generator = self.task.build_generator([model], gen_args)

    def _run_generator(self, model, base_sample):
        with torch.no_grad():
            hypos = self.generator.generate([model], base_sample)
        return hypos
    
    def _get_rewards(self, hypos, targets):
        rewards = []
        for hypo, rtarget in zip(hypos, targets):
            rewards.append([])
            ref = self._decode(rtarget, is_hyp=False)
            if not isinstance(hypo, list):
                hypo = [hypo]
            for preds in hypo:
                hyp = self._decode(preds)
                if self.task.cfg.eval_tokenized_bleu:
                    rewards[-1].append(sacrebleu.corpus_bleu([hyp], [[ref]], tokenize="none").score)
                else:
                    rewards[-1].append(sacrebleu.corpus_bleu([hyp], [[ref]]).score)
        return rewards

    def get_batch(self, model, base_sample):
        if (isinstance(self.task, TranslationWithActorCriticOffline)
            or not torch.is_grad_enabled() or self.learn_imitate):
            # either offline RL / imitation, or
            # validtion: we do not produce additional samples for validation.
            target = base_sample["target"]
            #gt_rewards = [[100]] * len(target) = self._get_rewards(target, target)
            if self.use_gold_reward:
                net_output = self.task.base_model(**base_sample["net_input"])
                lprobs = self.task.base_model.get_normalized_probs(net_output, log_probs=True)
                lprobs = torch.clamp(lprobs, -2.3)
                lprobs = torch.gather(lprobs, 2, target[:, :, None])
                reverse_cumsum = lprobs + torch.sum(lprobs, dim=1, keepdims=True) - torch.cumsum(lprobs, dim=1)
                rewards = reverse_cumsum.squeeze(-1) / torch.arange(lprobs.shape[-2], 0, -1).cuda() + 3.3
            else:
                rewards = base_sample.get("score", [[100]] * len(target))
            return len(target), 1, rewards, base_sample["net_input"], target
        with torch.no_grad():
            if self.generator is None:
                self._build_generator(model)
            hypos = self._run_generator(model, base_sample)
            hypos = [[preds["tokens"] for preds in each] for each in hypos]
            rewards = self._get_rewards(hypos, base_sample["target"])

            # append ground truth hypo and reward to training set
            # should only done while off-policy training
            if not self.use_reinforce and not self.use_ac:
                gt_rewards = self._get_rewards(base_sample["target"], base_sample["target"])
                for i in range(len(rewards)):
                    rewards[i] = rewards[i] + gt_rewards[i]
                    hypos[i] = hypos[i] + [base_sample["target"][i]]
                
            num_hypos = len(hypos)
            num_samples = len(hypos[0])

            hypos = [item for sublist in hypos for item in sublist]
            vinputs = {
                "src_tokens": base_sample["net_input"]["src_tokens"].tile(
                1, num_samples).view(num_hypos * num_samples, -1),
                "src_lengths": base_sample["net_input"]["src_lengths"][:, None].tile(
                    1, num_samples).view(num_hypos * num_samples)}
            vtargets = collate_tokens(hypos, self.pad_idx, self.eos_idx,
                left_pad=self.task.cfg.left_pad_target)
            vinputs["prev_output_tokens"] = collate_tokens(
                hypos, self.pad_idx, self.eos_idx,
                left_pad=self.task.cfg.left_pad_target,
                move_eos_to_beginning=True)
        del hypos
        return num_hypos, num_samples, rewards, vinputs, vtargets

    def _get_critic_loss(self, value, target_values, vtargets, shape, rewards, step_reward, without_residual_loss, do_not_clone, update_num):
        #value = model.vf(out_features.detach()) if self.detach_actor else model.vf(out_features)
        cur_v_star = torch.logsumexp(value, dim=2, keepdim=True).view(*shape)
        cur_q = torch.gather(value, 2, vtargets[:, :, None]).view(*shape)

        if self.use_pcl:
            cur_q_target = torch.gather(target_values, 2, vtargets[:, :, None]).view(*shape).detach()
            next_q_target = torch.nn.functional.pad(cur_q_target, [0, 1])[:, :, 1:]
            residual = cur_q - step_reward - next_q_target
            # Path Consistent Learning
            log_pi = cur_q - cur_v_star
            cur_v_target = torch.logsumexp(target_values, dim=2, keepdim=True).view(*shape).detach()
            next_v_target = torch.nn.functional.pad(cur_v_target, [0, 1])[:, :, 1:]
            step_pcl_loss = (-cur_v_target + next_v_target + step_reward - log_pi) ** 2
            cum_log_pi = log_pi + torch.sum(log_pi, dim=2, keepdims=True) - torch.cumsum(log_pi, dim=2)
            traj_pcl_loss = ((-cur_v_target + rewards - cum_log_pi)) ** 2

            critic_loss = step_pcl_loss + traj_pcl_loss
            return step_pcl_loss, traj_pcl_loss

        if self.use_iql:
            cur_q_target = torch.gather(target_values, 2, vtargets[:, :, None]).view(*shape).detach()
            next_q_target = torch.nn.functional.pad(cur_q_target, [0, 1])[:, :, 1:]
            residual = cur_q - step_reward - next_q_target
            critic_loss = 0.5 * residual.pow(2) * torch.abs(self.tau - torch.less(residual, 0).float())
            if not do_not_clone:
                critic_loss = critic_loss +  self.alpha * (cur_v_star - cur_q)
            return critic_loss, None
        """
        if not self.learn_imitate:
            residual = rewards - cur_q
        else:
            next_v_star = torch.cat([cur_v_star[:, :, 1:], 0 * cur_v_star[:, :, :1]], axis=2)
            residual = cur_q - next_v_star
            if without_residual_loss:
                residual = residual * 0.
        """
        max_q, _ = torch.max(value, dim=2, keepdim=True)
        max_q = max_q.view(*shape)
        critic_loss = 0.5 * (rewards - cur_q).pow(2) * torch.abs(self.tau - torch.less(rewards - cur_q, 0).float())

        
        if update_num < 5000 or self.use_full:
            reg = self.alpha * (cur_v_star - cur_q)
        else:
            reg = self.alpha * (cur_v_star - cur_q) * torch.exp((cur_q - max_q)).detach()
        if self.use_reward_weighted:
            reg = reg * rewards
        return critic_loss, reg


    def _get_actor_loss(self, value, vtargets, shape, rewards, subtract_mean=False):
        cur_v_star = torch.logsumexp(value, dim=2, keepdim=True).view(*shape)
        cur_q = torch.gather(value, 2, vtargets[:, :, None]).view(*shape)
        if subtract_mean:
            rewards = rewards - rewards.mean(1, keepdim=True)
        if self.use_bc:
            return cur_v_star - cur_q
        iw = torch.clamp(torch.exp((cur_q - cur_v_star)).detach(), min=0.1)
        return (cur_v_star - cur_q) * (rewards.detach() / 100) * iw

    def _average_loss(self, loss, non_pad_mask, batch_tokens):
        return torch.sum(loss[non_pad_mask]) / batch_tokens

    def _get_pad_mask(self, vtargets, shape):
        non_pad_mask = vtargets.ne(self.pad_idx).view(*shape)
        batch_tokens = non_pad_mask.sum() / shape[1]
        return non_pad_mask, batch_tokens

    def forward(self, model, sample, update_num=0, reduce=True, do_not_clone=False, without_residual_loss=False):

        num_hypos, num_samples, rewards, vinputs, vtargets = self.get_batch(model, sample)

        shape = [num_hypos, num_samples, -1]
        non_pad_mask, batch_tokens = self._get_pad_mask(vtargets, shape)

        if not torch.is_tensor(rewards):
            rewards = torch.Tensor(rewards).cuda()
        rewards = rewards.view(*shape)
        if not self.use_gold_reward:
            rewards = rewards / self.reward_scaler
        step_reward = (non_pad_mask.float() - torch.nn.functional.pad(non_pad_mask.float(), [0, 1])[:, :, 1:]) * rewards
        
        values = model(**vinputs)[0]
        if self.use_iql or self.use_pcl:
            target_values = self.task.target_net(**vinputs)[0]
        else:
            target_values = None

        avg_reg_loss = torch.Tensor([0.0]).cuda()
        if self.use_reinforce:
            policy_loss = self._get_actor_loss(values, vtargets, shape, rewards)
            avg_rl_loss = self._average_loss(policy_loss, non_pad_mask, batch_tokens)
        elif self.use_ac:
            critic_loss, cur_q = self._get_critic_loss(model, out_features, vtargets, shape, rewards, without_residual_loss)
            policy_loss = self._get_actor_loss(model, out_features, vtargets, shape, cur_q, subtract_mean=False)
            avg_rl_loss = self._average_loss(policy_loss, non_pad_mask, batch_tokens)
            avg_rl_loss += self._average_loss(critic_loss, non_pad_mask, batch_tokens)
        else:
            critic_loss, reg_loss = self._get_critic_loss(
                values, target_values, vtargets, shape, rewards, step_reward, without_residual_loss, do_not_clone, update_num)
            avg_rl_loss = self._average_loss(critic_loss, non_pad_mask, batch_tokens)
            avg_reg_loss = self._average_loss(reg_loss, non_pad_mask, batch_tokens)

        avg_tot_loss = avg_rl_loss + avg_reg_loss
        for param, target_param in zip(
            model.parameters(), self.task.target_net.parameters()):
            target_param.data.copy_(
                (1 - self.target_learning_rate) * target_param + 
                self.target_learning_rate * param
                )

        logging_output = {
            'loss': utils.item(avg_tot_loss.data),
            'rl_loss': utils.item(avg_rl_loss.data),
            'reg_loss': utils.item(avg_reg_loss.data),
            'ntokens': batch_tokens,
        }
        del vtargets, vinputs, rewards
        return avg_tot_loss, batch_tokens, logging_output

    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        loss = sum(log.get("loss", 0) for log in logging_outputs)
        rl_loss = sum(log.get("rl_loss", 0) for log in logging_outputs)
        reg_loss = sum(log.get("reg_loss", 0) for log in logging_outputs)
        metrics.log_scalar("loss", loss, ntokens)
        metrics.log_scalar("rl_loss", rl_loss, ntokens)
        metrics.log_scalar("reg_loss", reg_loss, ntokens)
        
    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
