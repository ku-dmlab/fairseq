

from dataclasses import dataclass, field

from fairseq import metrics, utils
from fairseq.criterions import register_criterion
from fairseq.criterions.actor_critic import ActorCriticCriterion, ActorCriticCriterionConfig
from fairseq.tasks.translation_with_actor_critic_post_edit_offline import TranslationWithActorCriticPostEditOffline

@dataclass
class ActorCriticPostEditCriterionConfig(ActorCriticCriterionConfig):
    base_reward_scale: float = field(default=0.5)
    use_base_residual: bool = field(default=True)
    use_mt_residual: bool = field(default=True)
    use_pe_residual: bool = field(default=True)

    clone_base: bool = field(default=False)
    clone_offline: bool = field(default=False)
    clone_pe: bool = field(default=False)
    clone_mt: bool = field(default=False)

@register_criterion(
    "actor_critic_post_edit", dataclass=ActorCriticPostEditCriterionConfig
)
class ActorCriticPostEditCriterion(ActorCriticCriterion):
    def __init__(self, task, cfg):
        self.cfg = cfg
        super().__init__(task, cfg)

    def forward(self, model, sample, reduce=True):
        #assert not self.learn_imitate and self.use_clone_loss
        if self.task.cfg.use_pe_for_eval and "base" in sample:
            if isinstance(self.task, TranslationWithActorCriticPostEditOffline):
                base_ret = super().forward(
                    model, sample["base"], reduce, reward_scaler=self.cfg.base_reward_scale, 
                    do_not_clone=not self.cfg.clone_base)
                pe_ret = super().forward(
                    model, sample["pe"], reduce, do_not_clone=not self.cfg.clone_pe)
                mt_ret = super().forward(
                    model, sample["mt"], reduce, do_not_clone=not self.cfg.clone_mt)
                all_returns = [base_ret, pe_ret, mt_ret]
                if "offline" in sample:    
                    offline_ret = super().forward(
                        model, sample["offline"], reduce, reward_scaler=self.cfg.base_reward_scale,
                        do_not_clone=not self.cfg.clone_offline)
                    all_returns.append(offline_ret)
            else:
                base_ret = super().forward(
                    model, sample["base"], reduce, without_residual_loss=self.use_base_residual)
                pe_ret = super().forward(
                    model, sample["pe"], reduce, without_residual_loss=self.use_mt_residual)
                mt_ret = super().forward(
                    model, sample["mt"], reduce, without_residual_loss=self.use_pe_residual)
                all_returns = [base_ret, pe_ret, mt_ret]
            loss = sum([each[0] for each in all_returns])
            sample_sizes = sum([each[1] for each in all_returns])
            logging_outputs = [each[2] for each in all_returns]
            logging_output = {k:sum([each[k] for each in logging_outputs]) for k in logging_outputs[0].keys()}
            return loss, sample_sizes, logging_output
        elif "base" in sample:
            return super().forward(
                model, sample["base"], reduce, without_residual_loss=self.use_base_residual)
        else:
            return super().forward(model, sample, reduce)
