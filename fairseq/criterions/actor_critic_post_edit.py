

from dataclasses import dataclass, field

from fairseq import metrics, utils
from fairseq.criterions import register_criterion
from fairseq.criterions.actor_critic import ActorCriticCriterion, ActorCriticCriterionConfig

@dataclass
class ActorCriticPostEditCriterionConfig(ActorCriticCriterionConfig):
    pe_reward_scale: float = field(default=2.0)
    mt_reward_scale: float = field(default=0.5)
    use_base_residual: bool = field(default=True)
    use_mt_residual: bool = field(default=True)
    use_pe_residual: bool = field(default=True)

@register_criterion(
    "actor_critic_post_edit", dataclass=ActorCriticPostEditCriterionConfig
)
class ActorCriticPostEditCriterion(ActorCriticCriterion):
    def __init__(self, task, cfg):
        self.pe_reward_scale = cfg.pe_reward_scale
        self.mt_reward_scale = cfg.mt_reward_scale
        self.use_base_residual = cfg.use_base_residual
        self.use_mt_residual = cfg.use_mt_residual
        self.use_pe_residual = cfg.use_pe_residual
        super().__init__(task, cfg)

    def forward(self, model, sample, reduce=True, critic_only=False):
        #assert not self.learn_imitate and self.use_clone_loss
        if "base" in sample:
            base_ret = super().forward(
                model, sample["base"], reduce, critic_only, without_residual_loss=self.use_base_residual)
            pe_ret = super().forward(
                model, sample["base"], reduce, critic_only, reward_scaler=self.pe_reward_scale, without_residual_loss=self.use_mt_residual)
            mt_ret = super().forward(
                model, sample["base"], reduce, critic_only, reward_scaler=self.mt_reward_scale, without_residual_loss=self.use_pe_residual)
            all_returns = [base_ret, pe_ret, mt_ret]
            loss = sum([each[0] for each in all_returns])
            sample_sizes = sum([each[1] for each in all_returns])
            logging_outputs = [each[2] for each in all_returns]
            logging_output = {k:sum([each[k] for each in logging_outputs]) for k in logging_outputs[0].keys()}
            return loss, sample_sizes, logging_output
        else:
            return super().forward(model, sample, reduce, critic_only)
