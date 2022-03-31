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
BASE_TASK = "translation"
AC_TASK = "translation_with_actor_critic"
SAMPLES_DIR = os.path.join(BASE_DIR, "samples")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(SAMPLES_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


class Experiment():
    BASE_TRAIN_ARGS = [
        "data-bin/iwslt14.tokenized.de-en",
        "--arch", "transformer_iwslt_de_en",
        "--share-decoder-input-output-embed",
        "--optimizer", "adam",
        "--adam-betas", "(0.9, 0.98)",
        "--clip-norm", "0.0",
        "--lr-scheduler", "inverse_sqrt",
        "--warmup-updates", "4000",
        "--dropout", "0.3",
        "--weight-decay", "0.0001",
        "--eval-bleu",
        "--eval-bleu-args", '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}',
        "--eval-bleu-detok", "moses",
        "--eval-bleu-remove-bpe",
        "--eval-bleu-print-samples",
        "--best-checkpoint-metric", "bleu",
        "--maximize-best-checkpoint-metric",
        "--patience", "10"]
    BASE_TEST_ARGS = [
        "data-bin/iwslt14.tokenized.de-en",
        "--batch-size", "128",
        "--beam", "5",
        "--remove-bpe"
    ]

    def __init__(self, exp_id, seed,
                 train_args=None, test_args=None, base_model_path=None, task=None, max_tokens=4096):

        self.exp_id = exp_id
        self.output_dir = os.path.join(BASE_DIR, exp_id, str(seed))
        os.makedirs(self.output_dir, exist_ok=True)
        self.sample_dir = os.path.join(SAMPLES_DIR, exp_id, str(seed) + ".gen")
        self.seed = seed
        self.train_task = self.test_task = BASE_TASK
        self.train_args = train_args or []
        self.test_args = test_args or []

        self.train_args += ["--max-tokens", str(max_tokens)]
        if task is None and base_model_path is not None:
            self.train_args += ["--restore-file", base_model_path]
        elif base_model_path is not None:
            self.train_args += ["--restore-file", base_model_path, "--base-model-path", base_model_path]
            self.test_args += ["--base-model-path", base_model_path]
            self.train_task = self.test_task = task

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

    def run(self, try_different_ratio=False):
        # subprocess call is requried to train multiple models without interference
        # "python" can be changed according to the machine's python settings
        train_path = os.path.join(str(pathlib.Path().resolve()), "fairseq_cli", "train.py")
        subprocess.call([PYTHON_PATH, train_path] + self.get_train_args(), env=os.environ)

        def test(critic_mixt_ratio=None):
            test_path = os.path.join(str(pathlib.Path().resolve()), "fairseq_cli", "generate.py")
            os.makedirs(os.path.dirname(self.sample_dir), exist_ok=True)
            add_arg = []
            if critic_mixt_ratio is not None:
                add_arg = ["--critic-mix-ratio", str(critic_mixt_ratio)]
            with open(self.sample_dir, 'w') as f:
                subprocess.call([PYTHON_PATH, test_path] + self.get_test_args() + add_arg, env=os.environ, stdout=f)
            with open(self.sample_dir, "r") as f:
                score = f.readlines()[-1].split()[6].strip(',')
            return score
        
        if not try_different_ratio:
            return {self.exp_id: test()}
        else:
            ret_dict = {}
            for ratio in [5.0, 2.5, 1.0, 0.5, 0.25, 0.1, 0.05]:
                ret_dict[f"{self.exp_id}_{ratio}"] = test(ratio)
            return ret_dict


def run_online(i):
    all_scores = {}
    run_baseline(i, all_scores)
    run_reinforce_online(i, all_scores)
    run_ours_online(i, all_scores)

    # save results
    print(all_scores)
    res_file = os.path.join(RESULTS_DIR, f"result_online_{i}.pkl")
    with open(res_file, "wb") as f:
        pickle.dump(all_scores, f)


def run_opt(i):
    all_scores = {}
    run_baseline(i, all_scores)
    opt_alpha(i, all_scores)
    opt_tau(i, all_scores)

    # save results
    print(all_scores)
    res_file = os.path.join(RESULTS_DIR, f"result_opt_{i}.pkl")
    with open(res_file, "wb") as f:
        pickle.dump(all_scores, f)

def opt_alpha(i, dict):
    alpha = {100:0, 101:0.1, 102:0.3, 103:3.0}
    run_ours_online(100, dict, alpha=alpha[i])

def opt_tau(i, dict):
    tau = {100:0.5, 101:0.75, 102:0.95, 103:0.97}
    run_ours_online(100, dict, tau=tau[i])

def run_baseline(i, dict):
    id = "baseline"
    train_args = ["--lr", "5e-4", "--criterion", "label_smoothed_cross_entropy_post_edit",
        "--label-smoothing", "0.1", "--use-base-for-train", "--max-epoch", "40"]
    exp = Experiment(id, i, train_args=train_args)
    dict.update(exp.run())

def run_reinforce_online(i, dict, use_clone_loss=True, use_beam_while_training=False, reward_scaler=50):
    base_model_path = os.path.join(BASE_DIR, "baseline", str(i), "checkpoint_best.pt")
    
    id = "reinforce_online"
    train_args = ["--lr", "5e-5", "--criterion", "actor_critic_post_edit", "--use-reinforce", "--reset-optimizer"]
    if use_clone_loss:
        train_args.append("--use-clone-loss")
        id += "_clone"
    if use_beam_while_training:
        train_args.append("--use-beam-while-training")
        id += "_beam"
    train_args.extend(["--reward-scaler", str(reward_scaler)])
    id += f"_reward_{reward_scaler}"
    
    exp = Experiment(id, i, train_args=train_args, task=AC_TASK, base_model_path=base_model_path, max_tokens=2048)
    dict.update(exp.run())

def run_ours_online(i, dict, use_clone_loss=True, use_beam_while_training=True, reward_scaler=50, alpha=None, tau=None):
    base_model_path = os.path.join(BASE_DIR, "baseline", str(i), "checkpoint_best.pt")
    
    id = "ours_online"
    train_args = ["--lr", "5e-5", "--criterion", "actor_critic_post_edit", "--reset-optimizer", "--use-critic-generator"]
    test_args = ["--use-critic-generator"]
    if use_clone_loss:
        train_args.append("--use-clone-loss")
        id += "_clone"
    if use_beam_while_training:
        train_args.append("--use-beam-while-training")
        train_args.extend(["--critic-mix-ratio", "0.5"])
        id += "_beam"
    if alpha is not None:
        train_args.extend(["--alpha", str(alpha)])
        id += f"_alpha_{alpha}"
    if tau is not None:
        train_args.extend(["--tau", str(tau)])
        id += f"_tau_{tau}"
    train_args.extend(["--reward-scaler", str(reward_scaler)])
    id += f"_reward_{reward_scaler}"
    
    exp = Experiment(id, i, train_args=train_args, test_args=test_args, task=AC_TASK, base_model_path=base_model_path)
    dict.update(exp.run(try_different_ratio=True))

def run_ac_online(i, dict, use_clone_loss=True, use_beam_while_training=False, reward_scaler=50):
    base_model_path = os.path.join(BASE_DIR, "baseline", str(i), "checkpoint_best.pt")
    
    id = "ac_online"
    train_args = ["--lr", "5e-5", "--criterion", "actor_critic_post_edit", "--use-ac", "--reset-optimizer"]
    if use_clone_loss:
        train_args.append("--use-clone-loss")
        id += "_clone"
    if use_beam_while_training:
        train_args.append("--use-beam-while-training")
        id += "_beam"
    train_args.extend(["--reward-scaler", str(reward_scaler)])
    id += f"_reward_{reward_scaler}"
    
    exp = Experiment(id, i, train_args=train_args, task=AC_TASK, base_model_path=base_model_path)
    dict.update(exp.run())


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]
    run_opt(100 + int(sys.argv[1]))
