#!/usr/bin/env python3 -u

if __name__ == "__main__":
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    from fairseq_cli.generate import main
    import sys
    from fairseq import options

    directory = "/ext2/bjlee/fairseq_ckpts/"
    base_model_path = "/ext/bjlee/fairseq_ckpts/supervised_14deen_35.28/checkpoint_best.pt"

    dirnames = ["amax_5e-5", "dmax_5e-5", "amean_5e-5", "dmean_5e-5"]
    mixture_ratios = [0.25]
    scores = {}
    for dirname in dirnames:
        scores[dirname] = {}
        for ratio in mixture_ratios:
            args = (["data-bin/iwslt14.tokenized.de-en"]
            + ["--task", "translation_with_actor_critic"]
            + ["--path", directory + dirname + "/checkpoint_best.pt"]
            + ["--base-model-path", base_model_path]
            + ["--batch-size", "128"]
            + ["--beam", "5"]
            + ["--remove-bpe"]
            + ["--use-critic-generator"]
            + ["--critic-mix-ratio", f"{ratio}"])
            sys.argv = [sys.argv[0]] + args
            parser = options.get_generation_parser()
            args = options.parse_args_and_arch(parser)
            scorer = main(args)
            scores[dirname][ratio] = scorer.score()
            print("score: ", scores)
    print("score: ", scores)
