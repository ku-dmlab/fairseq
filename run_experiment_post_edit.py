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
BASE_TASK = "translation"
AC_TASK = "translation_with_actor_critic"
SAMPLES_DIR = os.path.join(BASE_DIR, "samples")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(SAMPLES_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
training_critic_mix_ratio = 1


class Experiment():
    BASE_TRAIN_ARGS = [
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
        "--no-epoch-checkpoints",
        "--maximize-best-checkpoint-metric"]
    BASE_TEST_ARGS = [
        "--batch-size", "128",
        "--remove-bpe"
    ]

    def __init__(self, exp_id, seed,
                 train_args=None, test_args=None, base_model_path=None, task=None, max_tokens=4096, no_base_model=False,
                 data_path="data-bin/iwslt14.tokenized.en-de", test_only=False, do_not_restore=False):
        self.data_path = data_path
        self.exp_id = exp_id
        self.output_dir = os.path.join(BASE_DIR, exp_id, str(seed))
        os.makedirs(self.output_dir, exist_ok=True)
        self.sample_dir = os.path.join(SAMPLES_DIR, exp_id, str(seed) + ".gen")
        self.seed = seed
        self.train_task = task or BASE_TASK
        self.test_task = task or BASE_TASK
        self.train_args = train_args or []
        self.test_args = test_args or []
        self.test_only = test_only

        self.train_args += ["--max-tokens", str(max_tokens)]
        if base_model_path is not None:
            self.train_args += ["--validate-interval-updates", "4000"]
            if not do_not_restore:
                self.train_args += ["--restore-file", base_model_path]
            if self.train_task != BASE_TASK and not no_base_model:
                self.train_args += ["--base-model-path", base_model_path]
                self.test_args += ["--base-model-path", base_model_path]

    def get_train_args(self):
        dep_args = [
            "--seed", str(self.seed),
            "--task", self.train_task,
            "--save-dir", self.output_dir,
            "--log-file", os.path.join(self.output_dir, "log"),
            "--tensorboard-logdir", self.output_dir]
        return [self.data_path] + self.BASE_TRAIN_ARGS + dep_args + self.train_args

    def get_test_args(self):
        best_path = os.path.join(self.output_dir, "checkpoint_best.pt")
        last_path = os.path.join(self.output_dir, "checkpoint_last.pt")
        path = best_path if os.path.exists(best_path) else last_path
        dep_args = [
            "--seed", str(self.seed),
            "--task", self.test_task,
            "--path", path]
        return [self.data_path] + self.BASE_TEST_ARGS + dep_args + self.test_args

    def run(self, try_different_ratio=False, try_different_beam=False):
        # subprocess call is requried to train multiple models without interference
        # "python" can be changed according to the machine's python settings
        train_path = os.path.join(str(pathlib.Path().resolve()), "fairseq_cli", "train.py")
        if not self.test_only:
            subprocess.call([PYTHON_PATH, train_path] + self.get_train_args(), env=os.environ)

        def test(critic_mixt_ratio=None, beam=5):
            test_path = os.path.join(str(pathlib.Path().resolve()), "fairseq_cli", "generate.py")
            os.makedirs(os.path.dirname(self.sample_dir), exist_ok=True)
            add_arg = ["--beam", str(beam)]
            if critic_mixt_ratio is not None:
                add_arg = add_arg + ["--critic-mix-ratio", str(critic_mixt_ratio)]
            with open(self.sample_dir, 'w') as f:
                subprocess.call([PYTHON_PATH, test_path] + self.get_test_args() + add_arg, env=os.environ, stdout=f)
            with open(self.sample_dir, "r") as f:
                score = f.readlines()[-1].split()[6].strip(',')
            return score
        
        if not try_different_ratio:
            return {self.exp_id + "_1": test(beam=1), self.exp_id + "_5": test(beam=5), self.exp_id + "_10": test(beam=10)}
        elif try_different_beam:
            ret_dict = {}
            ret_dict[f"{self.exp_id}_1"] = test(1000.0, beam=1)
            for ratio in [1000.0, 5.0, 2.0, 1.0, 0.5, 0.25, 0.01]:
                for beam in [5, 10]:
                    ret_dict[f"{self.exp_id}_{ratio}_{beam}"] = test(ratio, beam=beam)
            return ret_dict
        else:
            ret_dict = {}
            for ratio in [5.0, 2.0, 1.0, 0.5]:
                ret_dict[f"{self.exp_id}_{ratio}"] = test(ratio)
            return ret_dict

def run_all(i, test_only=False):
    all_scores = {}
    run_baseline_half(i, all_scores, max_epoch=40, portion=0.5, test_only=test_only)
    run_baseline_wmt(i, all_scores, test_only=test_only)
    for remove_top in [0, 1, 2, 3]:
        run_offline_only(i, all_scores, model="baseline_40_0.5", max_epoch=50, use_bc=True, remove_top=remove_top, no_ensemble=True, test_only=test_only)
        run_offline_only(i, all_scores, model="baseline_40_0.5", max_epoch=50, use_reinforce=True, remove_top=remove_top, no_ensemble=True, test_only=test_only)
        run_offline_only(i, all_scores, model="baseline_40_0.5", max_epoch=50, alpha=100, tau=0.999, remove_top=remove_top, test_only=test_only)
        run_offline_only(i, all_scores, model="baseline_40_0.5", max_epoch=50, alpha=100, tau=0.999, use_cer_r=True, remove_top=remove_top, test_only=test_only)
    run_offline_only(i, all_scores, model="baseline_40_0.5", max_epoch=50, use_bc=True, use_mss=True, no_ensemble=True, test_only=test_only)
    run_offline_only(i, all_scores, model="baseline_40_0.5", max_epoch=50, use_reinforce=True, use_mss=True, no_ensemble=True, test_only=test_only)
    run_offline_only(i, all_scores, model="baseline_40_0.5", max_epoch=50, alpha=100, tau=0.999, use_mss=True, test_only=test_only)
    run_offline_only(i, all_scores, model="baseline_40_0.5", max_epoch=50, alpha=100, tau=0.999, use_cer_r=True, use_mss=True, test_only=test_only)

    for noise in [0, 1, 3, 5]:
       run_offline_only(i, all_scores, model="baseline_wmt", max_epoch=50, adapt=True, use_bc=True, noise=noise, no_ensemble=True, test_only=test_only)
       run_offline_only(i, all_scores, model="baseline_wmt", max_epoch=50, adapt=True, use_reinforce=True, noise=noise, no_ensemble=True, test_only=test_only)
       run_offline_only(i, all_scores, model="baseline_wmt", max_epoch=50, alpha=100, tau=0.999, adapt=True, noise=noise, no_ensemble=True, test_only=test_only)
       run_offline_only(i, all_scores, model="baseline_wmt", max_epoch=50, alpha=100, tau=0.999, adapt=True, use_cer_r=True, noise=noise, no_ensemble=True, test_only=test_only)
    run_offline_only(i, all_scores, model="baseline_wmt", max_epoch=50, use_bc=True, use_mss=True, adapt=True, no_ensemble=True, test_only=test_only)
    run_offline_only(i, all_scores, model="baseline_wmt", max_epoch=50, use_reinforce=True, use_mss=True, adapt=True, no_ensemble=True, test_only=test_only)
    run_offline_only(i, all_scores, model="baseline_wmt", max_epoch=50, alpha=100, tau=0.999, use_mss=True, adapt=True, test_only=test_only)
    run_offline_only(i, all_scores, model="baseline_wmt", max_epoch=50, alpha=100, tau=0.999, use_cer_r=True, use_mss=True, adapt=True, test_only=test_only)

    print(all_scores)

def run_baseline(i, dict, test_only=False, max_epoch=80):
    # Train in-domain baseline with full data (not used)
    id = f"baseline_{max_epoch}"
    train_args = ["--lr", "5e-4", "--criterion", "label_smoothed_cross_entropy",
        "--label-smoothing", "0.1", "--max-epoch", str(max_epoch)]
    exp = Experiment(id, i, train_args=train_args, test_only=test_only)
    dict.update(exp.run())

def run_baseline_half(i, dict, test_only=False, max_epoch=80, portion=0.5):
    # Train in-domain baseline with half data
    TASK = "translation_portion"
    id = f"baseline_{max_epoch}_{portion}"
    train_args = ["--lr", "5e-4", "--criterion", "label_smoothed_cross_entropy",
        "--label-smoothing", "0.1", "--max-epoch", str(max_epoch), "--portion", str(portion)]
    exp = Experiment(id, i, train_args=train_args, test_only=test_only, task=TASK)
    dict.update(exp.run())

def run_baseline_wmt(i, dict, test_only=False, data_path="data-bin/wmt17.en-de"):
    # Train cross-domain baseline
    id = "baseline_wmt"
    train_args = ["--lr", "5e-4", "--criterion", "label_smoothed_cross_entropy",
        "--label-smoothing", "0.1", "--max-epoch", "40"]
    exp = Experiment(id, i, train_args=train_args, data_path=data_path, test_only=test_only)
    dict.update(exp.run())

def run_offline_only(i, dict, model="baseline", test_only=False, adapt=False, use_reinforce=False, use_bc=False,
                     max_epoch=160, alpha=None, tau=None, use_cer_r=False,
                     noise=0, remove_top=0, use_mss=False, no_ensemble=False):
    base_model_path = os.path.join(BASE_DIR, model, str(i), "checkpoint_last.pt")
    TASK = "translation_with_actor_critic_offline"
    if use_mss:
        datasets = f"data-bin/wmt17.iwslt.mss_baseline_wmt_best_5_50_0_0.{i}.en-de" \
            if adapt else f"data-bin/iwslt.mss_baseline_40_0.5_best_5_50_0_0.{i}.en-de"
    else:
        datasets = f"data-bin/wmt17.iwslt.good_baseline_wmt_best_5_50_{noise}_{remove_top}.{i}.en-de" \
            if adapt else f"data-bin/iwslt.good_baseline_40_0.5_best_5_50_{noise}_{remove_top}.{i}.en-de"
   
    id = f"offline_only_{noise}_{remove_top}_{use_mss}_{model}_{max_epoch}_adapt_{adapt}"

    train_args = ["--lr", "1e-4", "--criterion", "actor_critic_offline", "--reset-optimizer", "--use-critic-generator",
                  "--offline-data", datasets, "--max-epoch", str(max_epoch), "--offline-only", "--use-beam-while-training",
                  "--critic-mix-ratio", str(1.0)]
    test_args = [] if no_ensemble else ["--use-critic-generator"]

    if alpha is not None:
        train_args.extend(["--alpha", str(alpha)])
        id += f"_alpha_{alpha}"
    if tau is not None:
        train_args.extend(["--tau", str(tau)])
        id += f"_tau_{tau}"
    
    if use_reinforce:
        train_args.append("--use-reinforce")
        id += f"_use_reinforce"
    if use_bc:
        train_args.append("--use-bc")
        id += f"_use_bc"
    if use_cer_r:
        train_args.append("--use-cer-r")
        id += f"_cer_r"
    data_path = "data-bin/wmt17.en-de.iwslt" if adapt else"data-bin/iwslt14.tokenized.en-de"
    exp = Experiment(id, i, train_args=train_args, test_args=test_args, base_model_path=base_model_path, task=TASK, data_path=data_path, test_only=test_only)
    dict.update(exp.run(try_different_ratio=not no_ensemble))

if __name__ == "__main__":
    sys.argv.append("1")
    os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]
    run_all(100 + int(sys.argv[1]))
