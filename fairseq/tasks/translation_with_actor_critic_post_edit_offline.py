from dataclasses import dataclass, field
import os
import logging
from collections import OrderedDict
import numpy as np
from tqdm import trange
import sacrebleu

from . import register_task
from .translation import load_langpair_dataset
from .translation_with_actor_critic import TranslationWithActorCritic, TranslationWithActorCriticConfig
from .translation_with_actor_critic_offline import ScoredLanguagePairDataset
from fairseq import utils
from fairseq.data import RoundRobinZipDatasets, Dictionary, data_utils

logger = logging.getLogger(__name__)

@dataclass
class TranslationWithActorCriticPostEditOfflineConfig(TranslationWithActorCriticConfig):
    mt_data: str = field(default="")
    pe_data: str = field(default="")
    use_pe_for_eval: bool = field(default=False)
    offline_data: str = field(default="")

@register_task("translation_with_actor_critic_post_edit_offline", dataclass=TranslationWithActorCriticPostEditOfflineConfig)
class TranslationWithActorCriticPostEditOffline(TranslationWithActorCritic):

    def _decode(self, toks, escape_unk=False):
        s = self.tgt_dict.string(
            toks.int().cpu(),
            self.cfg.eval_bleu_remove_bpe,
            unk_string=("UNKNOWNTOKENINREF" if escape_unk else "UNKNOWNTOKENINHYP"),
        )
        if self.tokenizer:
            s = self.tokenizer.decode(s)
        return s

    def build_mt_score(self, mt_dataset, pe_dataset, mt_score_path):
        scores = np.zeros(len(mt_dataset))
        for i in trange(len(mt_dataset)):
            mt_sample = mt_dataset[i]
            pe_sample = pe_dataset[i]
            hyp = self._decode(mt_sample["target"])
            ref = self._decode(pe_sample["target"])
            if self.cfg.eval_tokenized_bleu:
                scores[i] = sacrebleu.corpus_bleu([hyp], [ref], tokenize="none").score
            else:
                scores[i] = sacrebleu.corpus_bleu([hyp], [ref]).score
        np.save(mt_score_path, scores)

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

        if is_train:
            mt_score_path = os.path.join(self.cfg.mt_data, "score.npy")
            if not os.path.exists(mt_score_path):
                self.build_mt_score(mt_dataset, pe_dataset, mt_score_path)
            mt_dataset = ScoredLanguagePairDataset(mt_dataset, mt_score_path)
        data_dict = OrderedDict(base=base_dataset, mt=mt_dataset, pe=pe_dataset)
        
        if self.cfg.offline_data != "" and is_train:
            offline_dataset = get_langpair_dataset(self.cfg.offline_data)
            offline_dataset = ScoredLanguagePairDataset(
                offline_dataset, os.path.join(self.cfg.offline_data, "score.npy"))
            data_dict["offline"] = offline_dataset

        eval_key = "pe" if self.cfg.use_pe_for_eval else "base"
        self.datasets[split] = RoundRobinZipDatasets(
            data_dict, eval_key=None if is_train else eval_key)
