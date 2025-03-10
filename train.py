#!/usr/bin/env python3 -u
import os, sys, time

def run(lr=5e-5, tau=0.9, detached=False, seed=0, resume=False):
    from fairseq_cli.train import cli_main
    suffix_str = "d" if detached else "a"
    dirname = f"/ext2/bjlee/fairseq_ckpts/{suffix_str}_{lr}_{tau}_{seed}"
    os.makedirs(dirname, exist_ok=True)
    restore = dirname + "/checkpoint_best.pt" if resume else "/ext/bjlee/fairseq_ckpts/supervised_14deen_35.28/checkpoint_best.pt"
    args = (["data-bin/iwslt14.tokenized.de-en"]
    + ["--restore-file", restore]
    + ["--best-checkpoint-metric", "critic_loss"]
    + ["--tau", str(tau)]
    + ["--seed", str(seed + 12345)]
    + ["--patience", "100"]
    + ["--use-beam-while-training"]
    #+ ["--validate-interval-updates", "100"]
    + ["--arch", "transformer_iwslt_de_en"]
    + ["--share-decoder-input-output-embed"]
    + ["--reset-optimizer"]
    + ["--optimizer", "adam"]
    + ["--adam-betas", "(0.9, 0.98)"]
    + ["--clip-norm", "0.0"]
    + ["--lr", str(lr)]
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
    #+ ["--no-epoch-checkpoints"]
    + ["--save-interval", "3"]
    + ["--save-dir", dirname]
    + ["--log-file", dirname + "/log"]
    + ["--tensorboard-logdir", dirname])
    if detached:
        args = args + ["--detach-actor"]
    sys.argv = [sys.argv[0]] + args
    cli_main()

if __name__ == "__main__":
    print(sys.argv[1])
    os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]
    pid = int(sys.argv[1])
    #print(f"sleeping {1800 * pid} sec")
    #time.sleep(1800 * pid)
    taus = [0.7, 0.9, 0.95, 0.99]
    for seed in range(3):
        run(tau=taus[pid], seed=seed, resume = seed ==0)
