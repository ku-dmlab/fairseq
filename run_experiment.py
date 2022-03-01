#!/usr/bin/env python3 -u
import contextlib
import os
import pathlib
import pickle
import subprocess

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
        "data-bin/iwslt14.tokenized.de-en",
        "--batch-size", "128",
        "--beam", "5",
        "--remove-bpe"
    ]

    def __init__(self, output_dir, sample_dir, seed, train_task="translation", test_task="translation", train_args=None, test_args=None):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.sample_dir = sample_dir
        self.seed = seed
        self.train_task = train_task
        self.test_task = test_task
        self.train_args = train_args or []
        self.test_args = test_args or []

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
        python_path = "python"
        train_path = os.path.join(str(pathlib.Path().resolve()), "fairseq_cli", "train.py")
        subprocess.call([python_path, train_path] + self.get_train_args(), env=os.environ)

        python_path = "python"
        test_path = os.path.join(str(pathlib.Path().resolve()), "fairseq_cli", "generate.py")
        with open(self.sample_dir, 'w') as f:
            subprocess.call([python_path, test_path] + self.get_test_args(), env=os.environ, stdout=f)
        with open(self.sample_dir, "r") as f:
            score = f.readlines()[-1].split()[6].strip(',')
        return score

NUM_SEEDS = 3
BASE_DIR = "/ext2/bjlee/" # should be changed to the directory to store things
SAMPLES_DIR = os.path.join(BASE_DIR, "samples")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
BASELINE_DIR = os.path.join(BASE_DIR, "baseline")
os.makedirs(SAMPLES_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(BASELINE_DIR, exist_ok=True)

def train_baseline():
    all_scores = {}
    train_args = ["--lr", "5e-4", "--criterion", "label_smoothed_cross_entropy",
        "--label-smoothing", "0.1"]
    for i in range(NUM_SEEDS):
        output_dir = os.path.join(BASELINE_DIR, str(i))
        sample_dir = os.path.join(SAMPLES_DIR, f"baseline_{i}.gen")
        exp = Experiment(output_dir, sample_dir, i, train_args=train_args)
        all_scores[f"baseline_{i}"] = exp.run()
    print("---------------------------------------")
    print(all_scores)
    res_file = os.path.join(RESULTS_DIR, f"baseline_result.pkl")
    with open(res_file, "wb") as f:
        pickle.dump(all_scores, f)

def train_reinforce(offline=False):
    reinforce_dir = os.path.join(BASE_DIR, "reinforce")
    all_scores = {}
    train_args = ["--lr", "5e-5", "--criterion", "actor_critic",
        "--use-reinforce", "--use-clone-loss", "--reset-optimizer"]
    if offline:
        train_args.append("--learn-offline")
    for i in range(NUM_SEEDS):
        output_dir = os.path.join(reinforce_dir, str(i), str(offline))
        sample_dir = os.path.join(SAMPLES_DIR, f"reinforce_{offline}_{i}.gen")
        base_model_path = os.path.join(BASELINE_DIR, str(i), "checkpoint_best.pt")
        exp = Experiment(output_dir, sample_dir, i, train_task="translation_with_actor_critic", test_task="translation_with_actor_critic",
            train_args=train_args + ["--restore-file", base_model_path, "--base-model-path", base_model_path],
            test_args=["--base-model-path", base_model_path])
        all_scores[f"reinforce_{offline}_{i}"] = exp.run()
    print("---------------------------------------")
    print(all_scores)
    res_file = os.path.join(RESULTS_DIR, f"reinforce_{offline}_result.pkl")
    with open(res_file, "wb") as f:
        pickle.dump(all_scores, f)

def check_ours_alpha():
    alpha_dir = os.path.join(BASE_DIR, "alpha")
    all_scores = {}
    train_args = ["--lr", "5e-5", "--criterion", "actor_critic",
        "--use-clone-loss", "--reset-optimizer", "--learn-offline", "--use-critic-generator"]
    i = 0
    for alpha in [1e-2, 1e-1, 1.0, 1e1]:
        output_dir = os.path.join(alpha_dir, str(alpha))
        sample_dir = os.path.join(SAMPLES_DIR, f"alpha_{alpha}.gen")
        base_model_path = os.path.join(BASELINE_DIR, str(i), "checkpoint_best.pt")
        exp = Experiment(output_dir, sample_dir, i, train_task="translation_with_actor_critic", test_task="translation_with_actor_critic",
            train_args=train_args + ["--restore-file", base_model_path, "--base-model-path", base_model_path, "--alpha", str(alpha)],
            test_args=["--base-model-path", base_model_path])
        all_scores[f"alpha_{alpha}"] = exp.run()
    print("---------------------------------------")
    print(all_scores)
    res_file = os.path.join(RESULTS_DIR, f"alpha_result.pkl")
    with open(res_file, "wb") as f:
        pickle.dump(all_scores, f)

def check_ours_tau():
    tau_dir = os.path.join(BASE_DIR, "tau")
    all_scores = {}
    train_args = ["--lr", "5e-5", "--criterion", "actor_critic",
        "--use-clone-loss", "--reset-optimizer", "--learn-offline", "--use-critic-generator"]
    i = 0
    for tau in [0.5, 0.7, 0.9, 0.95, 0.99]:
        output_dir = os.path.join(tau_dir, str(tau))
        sample_dir = os.path.join(SAMPLES_DIR, f"tau_{tau}.gen")
        base_model_path = os.path.join(BASELINE_DIR, str(i), "checkpoint_best.pt")
        exp = Experiment(output_dir, sample_dir, i, train_task="translation_with_actor_critic", test_task="translation_with_actor_critic",
            train_args=train_args + ["--restore-file", base_model_path, "--base-model-path", base_model_path, "--tau", str(tau)],
            test_args=["--base-model-path", base_model_path])
        all_scores[f"tau_{tau}"] = exp.run()
    print("---------------------------------------")
    print(all_scores)
    res_file = os.path.join(RESULTS_DIR, f"tau_result.pkl")
    with open(res_file, "wb") as f:
        pickle.dump(all_scores, f)

def train_ours(offline=False, imitate=False):
    ours_dir = os.path.join(BASE_DIR, "ours")
    all_scores = {}
    train_args = ["--lr", "5e-5", "--criterion", "actor_critic",
        "--use-clone-loss", "--reset-optimizer", "--use-critic-generator"]
    if offline:
        train_args.append("--learn-offline")
    if imitate:
        train_args.append("--learn-imitate")
    for i in range(NUM_SEEDS):
        output_dir = os.path.join(ours_dir, str(i), str(offline), str(imitate))
        sample_dir = os.path.join(SAMPLES_DIR, f"ours_{i}_{offline}_{imitate}.gen")
        base_model_path = os.path.join(BASELINE_DIR, str(i), "checkpoint_best.pt")
        exp = Experiment(output_dir, sample_dir, i, train_task="translation_with_actor_critic", test_task="translation_with_actor_critic",
            train_args=train_args + ["--restore-file", base_model_path, "--base-model-path", base_model_path],
            test_args=["--base-model-path", base_model_path])
        all_scores[f"ours_{i}_{offline}_{imitate}"] = exp.run()
    print("---------------------------------------")
    print(all_scores)
    res_file = os.path.join(RESULTS_DIR, f"ours_result_{offline}_{imitate}.pkl")
    with open(res_file, "wb") as f:
        pickle.dump(all_scores, f)

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # experiment with multiple gpus is not validated enough. Recommend running experiments with single GPU setting
    # also, this scripts may malfunction if there is already a trained model in current directory.
    # remove them before running.

    train_baseline()
    train_reinforce()
    train_reinforce(offline=True)
    check_ours_alpha()
    check_ours_tau()
    train_ours(offline=False)
    train_ours(offline=True)
    train_ours(imitate=True)

