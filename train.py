#!/usr/bin/env python3 -u

device = "1"

if __name__ == "__main__":
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = device
    from fairseq_cli.train import cli_main
    import sys

    lr = "5e-5"
    use_expectile_regression = True
    detached = False

    suffix_str = "d" if detached else "a"
    name_str = "mean" if use_expectile_regression else "max"
    tau = "0.9" if use_expectile_regression else "0.5"
    dirname = f"/ext2/bjlee/fairseq_ckpts/{suffix_str}{name_str}_{lr}"
    os.makedirs(dirname, exist_ok=True)
    args = (["data-bin/iwslt14.tokenized.de-en"]
    + ["--restore-file", "/ext/bjlee/fairseq_ckpts/supervised_14deen_35.28/checkpoint_best.pt"]
    + ["--best-checkpoint-metric", "critic_loss"]
    + ["--tau", tau]
    + ["--patience", "100"]
    + ["--use-beam-while-training"]
    + ["--validate-interval-updates", "100"]
    + ["--arch", "transformer_iwslt_de_en"]
    + ["--share-decoder-input-output-embed"]
    + ["--reset-optimizer"]
    + ["--optimizer", "adam"]
    + ["--adam-betas", "(0.9, 0.98)"]
    + ["--clip-norm", "0.0"]
    + ["--lr", lr]
    + ["--lr-scheduler", "inverse_sqrt"]
    + ["--warmup-updates", "4000"]
    + ["--dropout", "0.3"]
    + ["--weight-decay", "0.0001"]
    + ["--task", "translation_with_actor_critic"]
    + ["--criterion", "actor_critic"]
    + ["--max-tokens", "4096"]
    + ["--eval-bleu"]
    + ["--eval-bleu-args", '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}']
    + ["--eval-bleu-detok", "moses"]
    + ["--eval-bleu-remove-bpe"]
    + ["--eval-bleu-print-samples"]
    #+ ["--maximize-best-checkpoint-metric"]
    + ["--no-epoch-checkpoints"]
    + ["--save-dir", dirname]
    + ["--log-file", dirname + "/log"]
    + ["--tensorboard-logdir", dirname])
    if detached:
        args = args + ["--detach-actor"]
    sys.argv = [sys.argv[0]] + args
    cli_main()
