from dataclasses import dataclass, field
import os
import logging
import numpy as np
import torch
from collections import OrderedDict

from fairseq.data.language_pair_dataset import LanguagePairDataset

from . import register_task
from .translation import load_langpair_dataset
from .translation_with_actor_critic import TranslationWithActorCritic, TranslationWithActorCriticConfig
from fairseq import utils
from fairseq.data import RoundRobinZipDatasets, Dictionary, data_utils

logger = logging.getLogger(__name__)

@dataclass
class TranslationWithActorCriticOfflineConfig(TranslationWithActorCriticConfig):
    offline_data: str = field(default="")

@register_task("translation_with_actor_critic_offline", dataclass=TranslationWithActorCriticOfflineConfig)
class TranslationWithActorCriticOffline(TranslationWithActorCritic):

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
        if split == "train":
            base_dataset = get_langpair_dataset(self.cfg.data)
            offline_dataset = get_langpair_dataset(self.cfg.offline_data)
            offline_dataset = ScoredLanguagePairDataset(
                offline_dataset, os.path.join(self.cfg.offline_data, "score.npy"))

            self.datasets[split] = RoundRobinZipDatasets(
                OrderedDict(base=base_dataset, offline=offline_dataset))
        else:
            self.datasets[split] = get_langpair_dataset(self.cfg.data)

class ScoredLanguagePairDataset(LanguagePairDataset):
    def __init__(self, lang_pair_dataset, score_path):
        self.__dict__.update(lang_pair_dataset.__dict__)
        self.score = torch.FloatTensor(np.load(score_path))

    def __getitem__(self, index):
        example = super().__getitem__(index)
        example["score"] = self.score[index]
        return example

    def collater(self, samples):
        batch = super().collater(samples)
        batch["score"] = self.score[batch["id"]]
        return batch
