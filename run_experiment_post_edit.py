#!/usr/bin/env python3 -u
import contextlib
import itertools
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
    POST_EDIT_ARGS = [
        "--mt-data", "data-bin/iwslt14.tokenized.en-de.mt",
        "--pe-data", "data-bin/iwslt14.tokenized.en-de.pe"]
    BASE_TRAIN_ARGS = [
        "data-bin/iwslt14.tokenized.en-de",
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
        "--maximize-best-checkpoint-metric"]
    BASE_TEST_ARGS = [
        "data-bin/iwslt14.tokenized.en-de",
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
        self.train_task = task or BASE_TASK
        self.test_task = task or BASE_TASK
        self.train_args = train_args or []
        self.test_args = test_args or []

        self.train_args += ["--max-tokens", str(max_tokens)]
        if base_model_path is None:
            self.train_args += ["--patience", "10"]
        else:
            self.train_args += ["--patience", "100", "--validate-interval-updates", "100"]
            self.train_args += ["--restore-file", base_model_path]
            if self.train_task != BASE_TASK:
                self.train_args += ["--base-model-path", base_model_path]
                self.test_args += ["--base-model-path", base_model_path]
            if "post_edit" in self.train_task:
                self.train_args += self.POST_EDIT_ARGS
                self.test_args += self.POST_EDIT_ARGS

    def get_train_args(self):
        dep_args = [
            "--seed", str(self.seed),
            "--task", self.train_task,
            "--save-dir", self.output_dir,
            "--log-file", os.path.join(self.output_dir, "log"),
            "--tensorboard-logdir", self.output_dir]
        return self.BASE_TRAIN_ARGS + dep_args + self.train_args

    def get_test_args(self):
        best_path = os.path.join(self.output_dir, "checkpoint_best.pt")
        last_path = os.path.join(self.output_dir, "checkpoint_last.pt")
        path = best_path if os.path.exists(best_path) else last_path
        dep_args = [
            "--seed", str(self.seed),
            "--task", self.test_task,
            "--path", path]
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
    tau = {100:0.5, 101:0.75, 102:0.95, 103:0.99}
    run_ours_online(100, dict, tau=tau[i])

def run_opt_offline(i):
    all_scores = {}
    alpha = [0.0, 0.1, 0.3, 1.0, 3.0, 10.0]
    tau = [0.5, 0.75, 0.9, 0.95, 0.97, 0.99]
    for ind, (a, t) in enumerate(itertools.product(alpha, tau)):
        if ind % 4 == (i - 100):
            run_ours_offline(100, all_scores, alpha=a, tau=t)

    # save results
    print(all_scores)
    res_file = os.path.join(RESULTS_DIR, f"result_offline_opt_{i}.pkl")
    with open(res_file, "wb") as f:
        pickle.dump(all_scores, f)

def run_offline(i):
    all_scores = {}
    run_ours_offline(i, all_scores, alpha=1.0, tau=0.7)

    # save results
    print(all_scores)
    res_file = os.path.join(RESULTS_DIR, f"result_offline_{i}.pkl")
    with open(res_file, "wb") as f:
        pickle.dump(all_scores, f)

def run_imitate(i):
    all_scores = {}
    run_ours_imitate(i, all_scores, alpha=0.3)
    run_ours_imitate(i, all_scores, alpha=1.0)
    run_ours_imitate(i, all_scores, alpha=3.0)

    # save results
    print(all_scores)
    res_file = os.path.join(RESULTS_DIR, f"result_imitate_opt_{i}.pkl")
    with open(res_file, "wb") as f:
        pickle.dump(all_scores, f)

def run_su_adapt_all(i):
    all_scores = {}
    run_supervised_adaptation(i, all_scores, use_base=True)
    run_supervised_adaptation(i, all_scores, use_pe=True)
    run_supervised_adaptation(i, all_scores, use_mt=True)
    run_supervised_adaptation(i, all_scores, use_base=True, use_pe=True)
    run_supervised_adaptation(i, all_scores, use_base=True, use_mt=True)
    run_supervised_adaptation(i, all_scores, use_base=True, use_mt=True, use_pe=True)

    # save results
    print(all_scores)
    res_file = os.path.join(RESULTS_DIR, f"result_su_adapt_{i}.pkl")
    with open(res_file, "wb") as f:
        pickle.dump(all_scores, f)

def run_baseline(i, dict):
    id = "baseline"
    train_args = ["--lr", "5e-4", "--criterion", "label_smoothed_cross_entropy_post_edit",
        "--label-smoothing", "0.1", "--use-base-for-train", "--max-epoch", "40"]
    exp = Experiment(id, i, train_args=train_args)
    dict.update(exp.run())

def run_supervised_adaptation(i, dict, use_base=False, use_pe=False, use_mt=False):
    assert use_base or use_pe or use_mt
    base_model_path = os.path.join(BASE_DIR, "baseline", str(i), "checkpoint_best.pt")
    id = "su_adapt"
    train_args = ["--lr", "5e-4", "--criterion", "label_smoothed_cross_entropy_post_edit",
        "--label-smoothing", "0.1", "--use-pe-for-eval"]
    test_args = ["--use-pe-for-eval"]
    if use_base:
        train_args.append("--use-base-for-train")
        id += "_base"
    if use_pe:
        train_args.append("--use-pe-for-train")
        id += "_pe"
    if use_mt:
        train_args.append("--use-mt-for-train")
        id += "_mt"
    exp = Experiment(id, i, train_args=train_args, test_args=test_args, base_model_path=base_model_path)
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

def run_ours_offline(i, dict, use_clone_loss=True, use_beam_while_training=True, reward_scaler=50, alpha=None, tau=None):
    base_model_path = os.path.join(BASE_DIR, "baseline", str(i), "checkpoint_best.pt")
    TASK = "translation_with_actor_critic_offline"
    
    id = "ours_offline"
    train_args = ["--lr", "5e-5", "--criterion", "actor_critic_offline", "--reset-optimizer", "--use-critic-generator",
                  "--offline-data", f"data-bin/iwslt14.tokenized.offline.{i}.en-de"]
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
    
    exp = Experiment(id, i, train_args=train_args, test_args=test_args, task=TASK, base_model_path=base_model_path)
    dict.update(exp.run(try_different_ratio=True))

def run_ours_imitate(i, dict, use_clone_loss=True, use_beam_while_training=True, reward_scaler=50, alpha=None):
    base_model_path = os.path.join(BASE_DIR, "baseline", str(i), "checkpoint_best.pt")
    
    id = "ours_imitate"
    train_args = ["--lr", "5e-5", "--criterion", "actor_critic_post_edit", "--reset-optimizer", "--use-critic-generator", "--learn-imitate"]
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
    train_args.extend(["--reward-scaler", str(reward_scaler)])
    id += f"_reward_{reward_scaler}"
    
    exp = Experiment(id, i, train_args=train_args, test_args=test_args, task=AC_TASK, base_model_path=base_model_path)
    dict.update(exp.run(try_different_ratio=True))


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]
    run_su_adapt_all(100 + int(sys.argv[1]))
