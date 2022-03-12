

from dataclasses import dataclass, field

from fairseq import metrics, utils
from fairseq.criterions import register_criterion
from fairseq.criterions.label_smoothed_cross_entropy import LabelSmoothedCrossEntropyCriterionConfig, LabelSmoothedCrossEntropyCriterion

@dataclass
class LabelSmoothedCrossEntropyPostEditCriterionConfig(LabelSmoothedCrossEntropyCriterionConfig):
    use_base_for_train: bool = field(default=False)
    use_mt_for_train: bool = field(default=False)
    use_pe_for_train: bool = field(default=False)

@register_criterion(
    "label_smoothed_cross_entropy_post_edit", dataclass=LabelSmoothedCrossEntropyPostEditCriterionConfig
)
class LabelSmoothedCrossEntropyPostEditCriterion(LabelSmoothedCrossEntropyCriterion):
    def __init__(self, task, cfg):
        self.use_base_for_train = cfg.use_base_for_train
        self.use_mt_for_train = cfg.use_mt_for_train
        self.use_pe_for_train = cfg.use_pe_for_train
        assert self.use_base_for_train or self.use_mt_for_train or self.use_pe_for_train
        super().__init__(task, cfg.sentence_avg, cfg.label_smoothing, cfg.ignore_prefix_size, cfg.report_accuracy)

    def forward(self, model, sample, reduce=True):
        #assert not self.learn_imitate and self.use_clone_loss
        if "base" in sample:
            losses, sample_sizes, logging_outputs = [], [], []
            data_to_train = ["base"] if self.use_base_for_train else []
            data_to_train += ["mt"] if self.use_mt_for_train else []
            data_to_train += ["pe"] if self.use_pe_for_train else []
            for each in data_to_train:
                loss, sample_size, logging_output = super().forward(model, sample[each], reduce)
                losses.append(loss)
                sample_sizes.append(sample_size)
                logging_outputs.append(logging_output)
            logging_output = {k:sum([each[k] for each in logging_outputs]) for k in logging_outputs[0].keys()}

            return sum(losses), sum(sample_sizes), logging_output
        else:
            return super().forward(model, sample, reduce)
