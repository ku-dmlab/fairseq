
from fairseq.criterions import register_criterion
from fairseq.criterions.actor_critic import ActorCriticCriterion, ActorCriticCriterionConfig

@register_criterion(
    "actor_critic_offline", dataclass=ActorCriticCriterionConfig
)
class ActorCriticOfflineCriterion(ActorCriticCriterion):

    def forward(self, model, sample, update_num=0):
        if "base" not in sample:
            return super().forward(model, sample, update_num)
        base_ret = super().forward(
            model, sample["base"], update_num)
        offline_ret = super().forward(
            model, sample["offline"], update_num, do_not_clone=True)
        all_returns = [base_ret, offline_ret]

        loss = sum([each[0] for each in all_returns])
        sample_sizes = sum([each[1] for each in all_returns])
        logging_outputs = [each[2] for each in all_returns]
        logging_output = {k:sum([each[k] for each in logging_outputs]) for k in logging_outputs[0].keys()}

        return loss, sample_sizes, logging_output
