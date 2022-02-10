#!/usr/bin/env python3 -u
import contextlib
import io
import sys
import os
import pickle


if __name__ == "__main__":
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    from fairseq_cli.generate import main
    import sys
    from fairseq import options

    directory = "/ext2/bjlee/fairseq_ckpts/"
    base_model_path = "/ext/bjlee/fairseq_ckpts/supervised_14deen_35.28/checkpoint_best.pt"

    dirnames = ["a_5e-05_0.9_0_True"]
    mixture_ratios = [5.0, 1.0, 0.5, 0.25, 0.18, 0.12, 0.06, 0.03, 0]
    scores = {}
    for dirname in dirnames:
        scores[dirname] = {}
        cur_path = os.path.join(directory, dirname)
        ckpts = [f for f in os.listdir(cur_path)
                    if os.path.isfile(os.path.join(cur_path, f))]
        print(ckpts)
        for ckpt in ckpts:
            if ".pt" not in ckpt or "last" in ckpt or "best" in ckpt:
                continue
            scores[dirname][ckpt] = {}
            for ratio in mixture_ratios:
                args = (["data-bin/iwslt14.tokenized.de-en"]
                + ["--task", "translation_with_actor_critic"]
                + ["--path", os.path.join(cur_path, ckpt)]
                + ["--base-model-path", base_model_path]
                + ["--batch-size", "128"]
                + ["--beam", "5"]
                + ["--remove-bpe"]
                + ["--use-critic-generator"]
                + ["--critic-mix-ratio", f"{ratio}"])
                sys.argv = [sys.argv[0]] + args
                parser = options.get_generation_parser()
                args = options.parse_args_and_arch(parser)
                with open(os.devnull, 'w') as devnull:
                    with contextlib.redirect_stdout(devnull):
                        scorer = main(args)
                scores[dirname][ckpt][ratio] = scorer.score()
                print("score: ", scores)
    print("score: ", scores)
    with open("score2.pkl", "wb") as handle:
        pickle.dump(scores, handle)
