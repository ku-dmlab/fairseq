
import os
import sys
import shutil

def main(cfg, out_dir, src_pfx, tgt_pfx, data_key="base", num_samples=5, noise=1, remove_top=0):
    
    import ast
    import logging
    import numpy as np
    import torch
    from argparse import Namespace
    import sacrebleu

    from dataclasses import dataclass
    from fairseq import utils, options, tasks, checkpoint_utils
    from fairseq.data import Dictionary, data_utils, indexed_dataset, iterators
    from fairseq.dataclass.utils import convert_namespace_to_omegaconf
    from fairseq.file_io import PathManager
    from fairseq.logging import progress_bar

    tgt_bin = os.path.join(out_dir, f"{tgt_pfx}.bin")
    tgt_idx = os.path.join(out_dir, f"{tgt_pfx}.idx")
    
    src_bin = os.path.join(out_dir, f"{src_pfx}.bin")
    src_idx = os.path.join(out_dir, f"{src_pfx}.idx")
    score_bin = os.path.join(out_dir, "score.npy")
    dataset_impl = "mmap"

    if isinstance(cfg, Namespace):
        cfg = convert_namespace_to_omegaconf(cfg)
    cfg.dataset.gen_subset = "train"
    logger = logging.getLogger("build_offline_dataset")

    utils.import_user_module(cfg.common)

    if cfg.dataset.max_tokens is None and cfg.dataset.batch_size is None:
        cfg.dataset.max_tokens = 12000
    logger.info(cfg)
    
    if cfg.common.seed is not None and not cfg.generation.no_seed_provided:
        np.random.seed(cfg.common.seed)
        utils.set_torch_seed(cfg.common.seed)
    
    use_cuda = torch.cuda.is_available() and not cfg.common.cpu
    task = tasks.setup_task(cfg.task)
    src_dict = task.source_dictionary
    tgt_dict = task.target_dictionary
    
    overrides = ast.literal_eval(cfg.common_eval.model_overrides)
    logger.info("loading model(s) from {}".format(cfg.common_eval.path))
    models, saved_cfg = checkpoint_utils.load_model_ensemble(
        utils.split_paths(cfg.common_eval.path),
        arg_overrides=overrides,
        task=task,
        suffix=cfg.checkpoint.checkpoint_suffix,
        strict=(cfg.checkpoint.checkpoint_shard_count == 1),
        num_shards=cfg.checkpoint.checkpoint_shard_count,
    )

    task.load_dataset(cfg.dataset.gen_subset, task_cfg=saved_cfg.task)
    for model in models:
        if cfg.common.fp16:
            model.half()
        if use_cuda and not cfg.distributed_training.pipeline_model_parallel:
            model.cuda()
        model.prepare_for_inference_(cfg)
    if getattr(task.dataset(cfg.dataset.gen_subset), "datasets", None) is None:
        dataset = task.dataset(cfg.dataset.gen_subset)
    else:
        dataset = task.dataset(cfg.dataset.gen_subset).datasets[data_key]
    indices = np.arange(len(dataset), dtype=np.int64)
    batch_sampler = dataset.batch_by_size(
        indices, max_tokens=cfg.dataset.max_tokens, max_sentences=cfg.dataset.batch_size,
        required_batch_size_multiple=cfg.dataset.required_batch_size_multiple)
    itr = iterators.EpochBatchIterator(
            dataset=dataset,
            collate_fn=dataset.collater,
            batch_sampler=batch_sampler,
            seed=cfg.common.seed,
            num_shards=cfg.distributed_training.distributed_world_size,
            shard_id=cfg.distributed_training.distributed_rank,
            num_workers=cfg.dataset.num_workers,
            epoch=1,
            buffer_size=cfg.dataset.data_buffer_size,
        ).next_epoch_itr(shuffle=False)

    progress = progress_bar.progress_bar(
        itr,
        log_format=cfg.common.log_format,
        log_interval=cfg.common.log_interval,
        default_log_format=("tqdm" if not cfg.common.no_progress_bar else "simple"),
    )
    generator = task.build_generator(models, cfg.generation)

    tgt_ds = indexed_dataset.make_builder(
        tgt_bin, impl=dataset_impl, vocab_size=len(tgt_dict)
    )
    src_ds = indexed_dataset.make_builder(
        src_bin, impl=dataset_impl, vocab_size=len(src_dict)
    )
    
    def decode(toks, is_hyp=True):
        if not is_hyp:
            toks = utils.strip_pad(toks, tgt_dict.pad())
        s = tgt_dict.string(toks.int().cpu(), task.cfg.eval_bleu_remove_bpe,
            unk_string=("UNKNOWNTOKENINHYP" if is_hyp else "UNKNOWNTOKENINREF"))
        if task.tokenizer:
            s = task.tokenizer.decode(s)
        return s
    def get_rewards(hypos, targets):
        rewards = []
        for hypo, rtarget in zip(hypos, targets):
            ref = decode(rtarget, is_hyp=False)
            hyp = decode(hypo)
            if task.cfg.eval_tokenized_bleu:
                rewards.append(sacrebleu.corpus_bleu([hyp], [[ref]], tokenize="none").score)
            else:
                rewards.append(sacrebleu.corpus_bleu([hyp], [[ref]]).score)
        return rewards

    scores = []
    for sample in progress:
        sample = utils.move_to_cuda(sample) if use_cuda else sample
        if "net_input" not in sample:
            sample = sample[data_key]
        sorted_ind = sample["id"].sort().indices
        hypos = task.inference_step(generator, models, sample)
        sorted_targets = sample["target"][sorted_ind]
        sorted_sources = sample["net_input"]["src_tokens"][sorted_ind].cpu()

        all_sorted_hypo_tokens = []
        all_rewards = []
        for i in range(len(hypos[0])):
            sorted_hypo_tokens = [hypos[ind][i]["tokens"].cpu() for ind in sorted_ind]
            all_sorted_hypo_tokens.append(sorted_hypo_tokens)
            all_rewards.append(get_rewards(sorted_hypo_tokens, sorted_targets))
        all_rewards = np.array(all_rewards)
        all_rewards +=  np.random.randn(*all_rewards.shape) * noise
        good_ind = np.argsort(all_rewards, axis=0)[::-1][remove_top:num_samples + remove_top]
        rewards = np.take_along_axis(all_rewards, good_ind, 0)
        all_sorted_hypo_tokens = np.take_along_axis(np.array(all_sorted_hypo_tokens), good_ind, 0)
        
        scores += list(rewards.ravel())
        for i in range(num_samples):
            for src, hypo in zip(sorted_sources, all_sorted_hypo_tokens[i]):
                src_ds.add_item(src)
                tgt_ds.add_item(hypo)
    scores = np.array(scores)
    print("average_scores: ", np.average(scores))
    src_ds.finalize(src_idx)
    tgt_ds.finalize(tgt_idx)
    np.save(score_bin, np.array(scores))
    return np.average(scores)

def get_cfg_and_run(seed, epoch="_best", model="baseline_wmt", num_samples=5, beam_size=50, noise=0, adapt=False, remove_top=0):
    from fairseq import options
    
    if adapt:
        OUT_DIR = f"data-bin/wmt17.iwslt.good_{model}{epoch}_{num_samples}_{beam_size}_{noise}_{remove_top}.{seed}.en-de"
        BASE_DATA_DIR = "data-bin/wmt17.en-de.iwslt"
    else:
        OUT_DIR = f"data-bin/iwslt.good_{model}{epoch}_{num_samples}_{beam_size}_{noise}_{remove_top}.{seed}.en-de"
        BASE_DATA_DIR = "data-bin/iwslt14.tokenized.en-de"
    SRC_PFX = "train.en-de.en"
    TGT_PFX = "train.en-de.de"
    BASE_DIR = "/home/bjlee/mtrl_exps/"
    BASE_TEST_ARGS = [
        BASE_DATA_DIR,
        "--beam", str(beam_size),
        "--nbest", str(beam_size),
        "--batch-size", "32",
        "--max-len-a", "1.2",
        "--max-len-b", "10",
        #"--sampling",
        "--remove-bpe",
        "--eval-bleu",
        "--eval-bleu-args", '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}',
        "--eval-bleu-detok", "moses",
        "--eval-bleu-remove-bpe",
        "--eval-bleu-print-samples",
        "--seed", str(seed),
        "--task", "translation",
        "--path", os.path.join(BASE_DIR, model, str(seed), f"checkpoint{epoch}.pt")
    ]
    sys.argv = [sys.argv[0]] + BASE_TEST_ARGS
    parser = options.get_generation_parser()
    args = options.parse_args_and_arch(parser)

    assert not os.path.exists(OUT_DIR), f"{OUT_DIR} already exists!"
    os.makedirs(OUT_DIR, exist_ok=True)
    files_to_copy = ["dict.de.txt", "dict.en.txt"]
    for fname in files_to_copy:
        shutil.copy(
            os.path.join(BASE_DATA_DIR, fname),
            os.path.join(OUT_DIR, fname))

    return main(args, out_dir=OUT_DIR, src_pfx=SRC_PFX, tgt_pfx=TGT_PFX, num_samples=num_samples, noise=noise, remove_top=remove_top)


if __name__ == "__main__":
    argv = sys.argv[1]
    os.environ["CUDA_VISIBLE_DEVICES"] = argv
    #r1 = get_cfg_and_run(100 + int(argv), model="baseline_40")
    r1 = get_cfg_and_run(100 + int(argv), model="baseline_40_0.5", noise=0, remove_top=4)
    #r1 = get_cfg_and_run(100 + int(argv), model="baseline_wmt", adapt=True, noise=3)
    #sys.argv = [sys.argv[0], argv]
    #r1 = get_cfg_and_run(100 + int(argv), model="baseline_wmt", adapt=True, noise=5)
    print(r1)
