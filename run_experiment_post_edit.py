#!/usr/bin/env python3 -u
import contextlib
import os
import pathlib
import pickle
import subprocess
import sys

PYTHON_PATH = "/home/bjlee/miniconda3/bin/python"
PROJECT_PATH = "/home/bjlee/fairseq/"
NUM_SEEDS = 1
BASE_DIR = "/home/bjlee/mtrl_exps/" # should be changed to the directory to store things
BASE_TASK = "translation_with_post_edit"
AC_TASK = "translation_with_actor_critic_post_edit"
SAMPLES_DIR = os.path.join(BASE_DIR, "samples")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(SAMPLES_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


class Experiment():
    BASE_TRAIN_ARGS = [
        "data-bin/iwslt14.tokenized.en-de",
        "--mt-data", "data-bin/iwslt14.tokenized.en-de.mt",
        "--pe-data", "data-bin/iwslt14.tokenized.en-de.pe",
        "--arch", "transformer_iwslt_de_en",
        "--share-decoder-input-output-embed",
        "--optimizer", "adam",
        "--adam-betas", "(0.9, 0.98)",
        "--clip-norm", "0.0",
        "--lr-scheduler", "inverse_sqrt",
        "--warmup-updates", "4000",
        "--dropout", "0.3",
        "--weight-decay", "0.0001",
        "--max-tokens", "2048", # If OOM happens, we should reduce this into half or sth. It will reduce training speed though.
        "--eval-bleu",
        "--eval-bleu-args", '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}',
        "--eval-bleu-detok", "moses",
        "--eval-bleu-remove-bpe",
        "--eval-bleu-print-samples",
        "--best-checkpoint-metric", "bleu",
        "--maximize-best-checkpoint-metric",
        "--patience", "10",
        "--no-epoch-checkpoints",
        "--save-interval", "5"]
    BASE_TEST_ARGS = [
        "data-bin/iwslt14.tokenized.en-de",
        "--mt-data", "data-bin/iwslt14.tokenized.en-de.mt",
        "--pe-data", "data-bin/iwslt14.tokenized.en-de.pe",
        "--batch-size", "128",
        "--beam", "5",
        "--remove-bpe"
    ]

    def __init__(self, exp_id, seed,
                 train_args=None, test_args=None, base_model_path=None, task=None):

        self.output_dir = os.path.join(BASE_DIR, exp_id, str(seed))
        os.makedirs(self.output_dir, exist_ok=True)
        self.sample_dir = os.path.join(SAMPLES_DIR, exp_id, str(seed) + ".gen")
        self.seed = seed
        self.train_task = self.test_task = BASE_TASK
        self.train_args = train_args or []
        self.test_args = test_args or []

        if task is None and base_model_path is not None:
            self.train_args += ["--restore-file", base_model_path]
        elif base_model_path is not None:
            self.train_args += ["--restore-file", base_model_path, "--base-model-path", base_model_path]
            self.test_args += ["--base-model-path", base_model_path]
            self.train_task = self.test_task = AC_TASK

    def get_train_args(self):
        dep_args = [
            "--seed", str(self.seed),
            "--task", self.train_task,
            "--save-dir", self.output_dir,
            "--log-file", os.path.join(self.output_dir, "log"),
            "--tensorboard-logdir", self.output_dir]
        return self.BASE_TRAIN_ARGS + dep_args + self.train_args

    def get_test_args(self):
        dep_args = [
            "--seed", str(self.seed),
            "--task", self.test_task,
            "--path", os.path.join(self.output_dir, "checkpoint_best.pt")]
        return self.BASE_TEST_ARGS + dep_args + self.test_args

    def run(self):
        # subprocess call is requried to train multiple models without interference
        # "python" can be changed according to the machine's python settings
        train_path = os.path.join(str(pathlib.Path().resolve()), "fairseq_cli", "train.py")
        subprocess.call([PYTHON_PATH, train_path] + self.get_train_args(), env=os.environ)
        
        test_path = os.path.join(str(pathlib.Path().resolve()), "fairseq_cli", "generate.py")
        os.makedirs(os.path.dirname(self.sample_dir), exist_ok=True)
        with open(self.sample_dir, 'w') as f:
            subprocess.call([PYTHON_PATH, test_path] + self.get_test_args(), env=os.environ, stdout=f)
        with open(self.sample_dir, "r") as f:
            score = f.readlines()[-1].split()[6].strip(',')
        return score


def get_output_dir(id, seed):
    return os.path.join(BASE_DIR, )

def experiment_single(i):
    all_scores = {}
    
    # train_baseline
    id = "baseline"
    train_args = ["--lr", "5e-4", "--criterion", "label_smoothed_cross_entropy_post_edit",
        "--label-smoothing", "0.1", "--use-base-for-train"]
    exp = Experiment(id, i, train_args=train_args)
    all_scores[id] = exp.run()

    base_model_path = os.path.join(BASE_DIR, id, str(i), "checkpoint_best.pt")
    
    # train_reinforce
    id = "reinforce"
    train_args = ["--lr", "5e-5", "--criterion", "actor_critic_post_edit",
        "--use-reinforce", "--use-clone-loss", "--reset-optimizer"]
    exp = Experiment(id, i, base_model_path=base_model_path, train_args=train_args)
    all_scores[id] = exp.run()

    # train_reinforce_offline
    id = "reinforce_offline"
    train_args.append("--learn-offline")
    exp = Experiment(id, i, base_model_path=base_model_path, train_args=train_args)
    all_scores[id] = exp.run()

    # ours_online
    id = "ours_online"
    train_args = ["--lr", "5e-5", "--criterion", "actor_critic_post_edit",
        "--use-clone-loss", "--reset-optimizer", "--use-critic-generator"]
    exp = Experiment(id, i, base_model_path=base_model_path, train_args=train_args)
    all_scores[id] = exp.run()

    # ours_offline
    id = "ours_offline"
    train_args.append("--learn-offline")
    exp = Experiment(id, i, base_model_path=base_model_path, train_args=train_args)
    all_scores[id] = exp.run()

    # ours_imitate
    id = "ours_imitate"
    train_args.append("--learn-imitate")
    exp = Experiment(id, i, base_model_path=base_model_path, train_args=train_args)
    all_scores[id] = exp.run()

    # save results
    res_file = os.path.join(RESULTS_DIR, f"result_{i}.pkl")
    with open(res_file, "wb") as f:
        pickle.dump(all_scores, f)

def experiment_pe_su(i):
    all_scores = {}
    id = "baseline"
    train_args = ["--lr", "5e-4", "--criterion", "label_smoothed_cross_entropy_post_edit",
        "--label-smoothing", "0.1", "--use-base-for-train"]
    exp = Experiment(id, i, train_args=train_args)
    all_scores[id] = exp.run()

    base_model_path = os.path.join(BASE_DIR, "baseline", str(i), "checkpoint_best.pt")
    test_args = ["--use-pe-for-eval"]
    
    id = "su_base"
    train_args = ["--lr", "5e-5", "--criterion", "label_smoothed_cross_entropy_post_edit",
        "--label-smoothing", "0.1", "--use-base-for-train", "--use-pe-for-eval", "--reset-optimizer"]
    exp = Experiment(id, i, base_model_path=base_model_path, train_args=train_args, test_args=test_args)
    all_scores[id] = exp.run()

    id = "su_pe"
    train_args = ["--lr", "5e-5", "--criterion", "label_smoothed_cross_entropy_post_edit",
        "--label-smoothing", "0.1", "--use-pe-for-train", "--use-pe-for-eval", "--reset-optimizer"]
    exp = Experiment(id, i, base_model_path=base_model_path, train_args=train_args, test_args=test_args)
    all_scores[id] = exp.run()

    id = "su_mt"
    train_args = ["--lr", "5e-5", "--criterion", "label_smoothed_cross_entropy_post_edit",
        "--label-smoothing", "0.1", "--use-mt-for-train", "--use-pe-for-eval", "--reset-optimizer"]
    exp = Experiment(id, i, base_model_path=base_model_path, train_args=train_args, test_args=test_args)
    all_scores[id] = exp.run()

    id = "su_base_pe"
    train_args = ["--lr", "5e-5", "--criterion", "label_smoothed_cross_entropy_post_edit",
        "--label-smoothing", "0.1", "--use-base-for-train", "--use-pe-for-train", "--use-pe-for-eval", "--reset-optimizer"]
    exp = Experiment(id, i, base_model_path=base_model_path, train_args=train_args, test_args=test_args)
    all_scores[id] = exp.run()

    id = "su_base_mt"
    train_args = ["--lr", "5e-5", "--criterion", "label_smoothed_cross_entropy_post_edit",
        "--label-smoothing", "0.1", "--use-base-for-train", "--use-mt-for-train", "--use-pe-for-eval", "--reset-optimizer"]
    exp = Experiment(id, i, base_model_path=base_model_path, train_args=train_args, test_args=test_args)
    all_scores[id] = exp.run()

    id = "su_pe_mt"
    train_args = ["--lr", "5e-5", "--criterion", "label_smoothed_cross_entropy_post_edit",
        "--label-smoothing", "0.1", "--use-pe-for-train", "--use-mt-for-train", "--use-pe-for-eval", "--reset-optimizer"]
    exp = Experiment(id, i, base_model_path=base_model_path, train_args=train_args, test_args=test_args)
    all_scores[id] = exp.run()

    id = "su_all"
    train_args = ["--lr", "5e-5", "--criterion", "label_smoothed_cross_entropy_post_edit",
        "--label-smoothing", "0.1", "--use-base-for-train", "--use-pe-for-train", "--use-mt-for-train", "--use-pe-for-eval", "--reset-optimizer"]
    exp = Experiment(id, i, base_model_path=base_model_path, train_args=train_args, test_args=test_args)
    all_scores[id] = exp.run()

    # save results
    res_file = os.path.join(RESULTS_DIR, f"result_pe_su_{i}.pkl")
    with open(res_file, "wb") as f:
        pickle.dump(all_scores, f)

def experiment_pe_rl(i):
    all_scores = {}

    base_model_path = os.path.join(BASE_DIR, "baseline", str(i), "checkpoint_best.pt")
    test_args = ["--use-pe-for-eval"]
    
    # train_reinforce
    id = "pe_reinforce"
    train_args = ["--lr", "5e-5", "--criterion", "actor_critic_post_edit",
        "--use-reinforce", "--use-clone-loss", "--reset-optimizer", "--use-pe-for-eval"]
    exp = Experiment(id, i, base_model_path=base_model_path, train_args=train_args, test_args=test_args)
    all_scores[id] = exp.run()

    # train_reinforce_offline
    id = "pe_reinforce_offline"
    train_args.append("--learn-offline")
    exp = Experiment(id, i, base_model_path=base_model_path, train_args=train_args)
    all_scores[id] = exp.run()

    # ours_online
    id = "pe_ours_online"
    train_args = ["--lr", "5e-5", "--criterion", "actor_critic_post_edit",
        "--use-clone-loss", "--reset-optimizer", "--use-critic-generator", "--use-pe-for-eval"]
    exp = Experiment(id, i, base_model_path=base_model_path, train_args=train_args)
    all_scores[id] = exp.run()

    # ours_offline
    id = "pe_ours_offline"
    train_args.append("--learn-offline")
    exp = Experiment(id, i, base_model_path=base_model_path, train_args=train_args)
    all_scores[id] = exp.run()

    # ours_imitate
    id = "pe_ours_imitate"
    train_args = ["--lr", "5e-5", "--criterion", "actor_critic_post_edit",
        "--use-clone-loss", "--reset-optimizer", "--use-critic-generator", "--use-pe-for-eval",
        "--learn-imitate", "--use-base-residual", "False", "--use-mt-residual", "False"]
    exp = Experiment(id, i, base_model_path=base_model_path, train_args=train_args)
    all_scores[id] = exp.run()

    # save results
    res_file = os.path.join(RESULTS_DIR, f"result_pe_rl_{i}.pkl")
    with open(res_file, "wb") as f:
        pickle.dump(all_scores, f)

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]
    experiment_pe_su(int(sys.argv[1]))
