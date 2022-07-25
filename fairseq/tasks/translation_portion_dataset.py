from dataclasses import dataclass, field
import logging

from fairseq.data.subsample_dataset import SubsampleDataset

from . import register_task
from .translation import TranslationConfig, TranslationTask

logger = logging.getLogger(__name__)

@dataclass
class TranslationPortionConfig(TranslationConfig):
    portion: float = field(default=0.5)

@register_task("translation_portion", dataclass=TranslationPortionConfig)
class TranslationPortion(TranslationTask):
    
    def __init__(self, cfg, src_dict, tgt_dict):
        super().__init__(cfg, src_dict, tgt_dict)
        self.portion = cfg.portion

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        super().load_dataset(split, epoch, combine, **kwargs)
        if split == "train":
            self.datasets[split] = SubsampleDataset(self.datasets[split], size_ratio=self.portion)
