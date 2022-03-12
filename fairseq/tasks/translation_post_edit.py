from dataclasses import dataclass, field
import os
import logging
from collections import OrderedDict

from . import register_task
from .translation import TranslationTask, TranslationConfig, load_langpair_dataset
from fairseq import utils
from fairseq.data import RoundRobinZipDatasets, data_utils

logger = logging.getLogger(__name__)

@dataclass
class TranslationWithPostEditConfig(TranslationConfig):
    mt_data: str = field(default="")
    pe_data: str = field(default="")
    use_pe_for_eval: bool = field(default=False)

@register_task("translation_with_post_edit", dataclass=TranslationWithPostEditConfig)
class TranslationWithPostEdit(TranslationTask):

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        # infer langcode
        src, tgt = self.cfg.source_lang, self.cfg.target_lang
        is_train = split == self.cfg.train_subset

        def get_langpair_dataset(dataset):
            paths = utils.split_paths(dataset)
            assert len(paths) > 0
            if not is_train:
                # if not training data set, use the first shard for valid and test
                paths = paths[:1]
            data_path = paths[(epoch - 1) % len(paths)]

            return load_langpair_dataset(
                data_path,
                split,
                src,
                self.src_dict,
                tgt,
                self.tgt_dict,
                combine=combine,
                dataset_impl=self.cfg.dataset_impl,
                upsample_primary=self.cfg.upsample_primary,
                left_pad_source=self.cfg.left_pad_source,
                left_pad_target=self.cfg.left_pad_target,
                max_source_positions=self.cfg.max_source_positions,
                max_target_positions=self.cfg.max_target_positions,
                load_alignments=self.cfg.load_alignments,
                truncate_source=self.cfg.truncate_source,
                num_buckets=self.cfg.num_batch_buckets,
                shuffle=(split != "test"),
                pad_to_multiple=self.cfg.required_seq_len_multiple,
            )
        base_dataset = get_langpair_dataset(self.cfg.data)
        mt_dataset = get_langpair_dataset(self.cfg.mt_data)
        pe_dataset = get_langpair_dataset(self.cfg.pe_data)

        eval_key = "pe" if self.cfg.use_pe_for_eval else "base"
        self.datasets[split] = RoundRobinZipDatasets(
            OrderedDict(base=base_dataset, mt=mt_dataset, pe=pe_dataset),
            eval_key=None if is_train else eval_key)

