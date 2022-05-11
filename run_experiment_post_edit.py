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
        "--beam", "5",
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
        if base_model_path is None:
            self.train_args += ["--patience", "3"]
        else:
            #self.train_args += ["--patience", "60", "--validate-interval-updates", "300"]
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

    def run(self, try_different_ratio=False):
        # subprocess call is requried to train multiple models without interference
        # "python" can be changed according to the machine's python settings
        train_path = os.path.join(str(pathlib.Path().resolve()), "fairseq_cli", "train.py")
        if not self.test_only:
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
            for ratio in [1000.0, 5.0, 2.0, 1.0, 0.5, 0.25, 0.01]:
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

def run_baseline_wmt(i, dict):
    id = "baseline_wmt"
    train_args = ["--lr", "5e-4", "--criterion", "label_smoothed_cross_entropy",
        "--label-smoothing", "0.1", "--max-epoch", "40"]
    exp = Experiment(id, i, train_args=train_args, data_path="data-bin/wmt17.en-de")
    dict.update(exp.run())

def run_su_adapt_all(i):
    all_scores = {}
    #run_baseline_wmt(i, all_scores)
    #run_supervised_adaptation(i, all_scores)
    # alpha search
    # tau search
    # reward scale search
    alpha = {100:0.1, 101:1.0, 102:10.0, 103:100.0}
    alpha = {100:0.0, 101:0.5, 102:1.0, 103:2.0}
    tau = {100:0.95, 101:0.97, 102:0.99, 103:0.999}
    
    def save_scores(ind):
        res_file = os.path.join(RESULTS_DIR, f"result_su_adapt_{i}_{ind}.pkl")
        with open(res_file, "wb") as f:
            pickle.dump(all_scores, f)

    #run_offline(i, all_scores, model="baseline_wmt", offline_data="su_adapt", alpha=1.0, tau=0.99, critic_mix_ratio=1000)
    run_offline(i, all_scores, model="baseline", alpha=1.0, tau=0.99, critic_mix_ratio=1000, adapt=False, do_not_restore=True)


    # save results
    print(all_scores)

def run_offline_adapt_all(i):
    all_scores = {}
    run_offline_adaptation(i, all_scores)
    run_offline_adaptation(i, all_scores, clone_pe=True)
    run_offline_adaptation(i, all_scores, clone_pe=True, clone_mt=True)
    # alpha = {100:0, 101:0.03, 102:0.1, 103:0.3}
    # run_offline_adaptation(100, all_scores, alpha=alpha[i], clone_pe=True)

    # save results
    print(all_scores)
    res_file = os.path.join(RESULTS_DIR, f"result_offline_adapt_{i}.pkl")
    with open(res_file, "wb") as f:
        pickle.dump(all_scores, f)

def run_baseline(i, dict):
    id = "baseline"
    train_args = ["--lr", "5e-4", "--criterion", "label_smoothed_cross_entropy_post_edit",
        "--label-smoothing", "0.1", "--use-base-for-train", "--max-epoch", "40"]
    exp = Experiment(id, i, train_args=train_args)
    dict.update(exp.run())

def run_supervised_adaptation(i, dict, use_wmt=False):
    base_model_path = os.path.join(BASE_DIR, "baseline_wmt", str(i), "checkpoint_best.pt")
    id = "su_adapt"
    train_args = ["--lr", "5e-4", "--criterion", "label_smoothed_cross_entropy",
        "--label-smoothing", "0.1"]
    data_path = "data-bin/wmt17.en-de.iwslt"
    if use_wmt:
        data_path = "data-bin/wmt17.en-de.iwslt:data-bin/wmt17.en-de"
        id += "_wmt"
    exp = Experiment(id, i, train_args=train_args, data_path=data_path, base_model_path=base_model_path)
    dict.update(exp.run())

def run_offline(i, dict, model="baseline_wmt", offline_data="baseline_wmt", alpha=None, tau=None, critic_mix_ratio=1000, use_pcl=False, 
            use_iql=False, test_only=False, adapt=True, do_not_restore=False):
    base_model_path = os.path.join(BASE_DIR, model, str(i), "checkpoint_best.pt")
    TASK = "translation_with_actor_critic_offline"

    # datasets = []
    # for beam_ind in range(beam_ind_end):
    #     datasets.append(f"data-bin/wmt17.iwslt.offline_{model}_best_{beam_ind}.{i}.en-de")
    # datasets = ",".join(datasets)
    datasets = f"data-bin/wmt17.iwslt.offline_{offline_data}.{i}.en-de" if adapt else f"data-bin/iwslt14.tokenized.offline.{i}.en-de"
    
    id = f"offline_real_final_{model}_adapt_{adapt}"
    if adapt:
        id += offline_data

    train_args = ["--lr", "5e-4", "--criterion", "actor_critic_offline", "--reset-optimizer", "--use-critic-generator",
                  "--offline-data", datasets, "--max-epoch", "150", "--use-beam-while-training", "--critic-mix-ratio", str(critic_mix_ratio)]
    test_args = ["--use-critic-generator"]

    if use_pcl:
        train_args.append("--use-pcl")
        id += f"_pcl"
    if use_iql:
        train_args.append("--use-iql")
        id += f"_iql"
    if alpha is not None:
        train_args.extend(["--alpha", str(alpha)])
        id += f"_alpha_{alpha}"
    if tau is not None:
        train_args.extend(["--tau", str(tau)])
        id += f"_tau_{tau}"

    data_path = "data-bin/wmt17.en-de.iwslt" if adapt else"data-bin/iwslt14.tokenized.en-de"
    exp = Experiment(id, i, train_args=train_args, test_args=test_args, task=TASK, base_model_path=base_model_path,
                     data_path=data_path, test_only=test_only, do_not_restore=do_not_restore)
    dict.update(exp.run(try_different_ratio=True))


def run_reinforce_online(i, dict, use_beam_while_training=False):
    base_model_path = os.path.join(BASE_DIR, "baseline", str(i), "checkpoint_best.pt")
    
    id = "reinforce_online"
    train_args = ["--lr", "5e-5", "--criterion", "actor_critic_post_edit", "--use-reinforce", "--reset-optimizer"]
    if use_beam_while_training:
        train_args.append("--use-beam-while-training")
        id += "_beam"
    
    exp = Experiment(id, i, train_args=train_args, task=AC_TASK, base_model_path=base_model_path, max_tokens=2048)
    dict.update(exp.run())

def run_ours_online(i, dict, use_beam_while_training=True, alpha=None, tau=None):
    base_model_path = os.path.join(BASE_DIR, "baseline", str(i), "checkpoint_best.pt")
    
    id = "ours_online"
    train_args = ["--lr", "5e-5", "--criterion", "actor_critic_post_edit", "--reset-optimizer", "--use-critic-generator"]
    test_args = ["--use-critic-generator"]
    if use_beam_while_training:
        train_args.append("--use-beam-while-training")
        train_args.extend(["--critic-mix-ratio", str(training_critic_mix_ratio)])
        id += "_beam"
    if alpha is not None:
        train_args.extend(["--alpha", str(alpha)])
        id += f"_alpha_{alpha}"
    if tau is not None:
        train_args.extend(["--tau", str(tau)])
        id += f"_tau_{tau}"
    
    exp = Experiment(id, i, train_args=train_args, test_args=test_args, task=AC_TASK, base_model_path=base_model_path)
    dict.update(exp.run(try_different_ratio=True))

def run_ours_offline(i, dict, epochs, use_beam_while_training=True, alpha=None, tau=None):
    base_model_path = os.path.join(BASE_DIR, "baseline_" + ",".join(epochs), str(i), "checkpoint_best.pt")
    TASK = "translation_with_actor_critic_offline"

    offline_data = [f"data-bin/iwslt14.tokenized.offline_{e}.{i}.en-de" for e in epochs]
    offline_data = ",".join(offline_data)

    id = "offline_nobase_" + ",".join(epochs)
    train_args = ["--lr", "5e-5", "--criterion", "actor_critic_offline", "--reset-optimizer", "--use-critic-generator",
                  "--offline-data", offline_data]
    test_args = ["--use-critic-generator"]
    if use_beam_while_training:
        train_args.append("--use-beam-while-training")
        train_args.extend(["--critic-mix-ratio", str(training_critic_mix_ratio)])
        id += "_beam"
    if alpha is not None:
        train_args.extend(["--alpha", str(alpha)])
        id += f"_alpha_{alpha}"
    if tau is not None:
        train_args.extend(["--tau", str(tau)])
        id += f"_tau_{tau}"
    
    exp = Experiment(id, i, train_args=train_args, test_args=test_args, task=TASK)#, base_model_path=base_model_path)
    dict.update(exp.run(try_different_ratio=True))

def run_ours_imitate(i, dict, use_beam_while_training=True, alpha=None):
    base_model_path = os.path.join(BASE_DIR, "baseline", str(i), "checkpoint_best.pt")
    
    id = "ours_imitate"
    train_args = ["--lr", "5e-5", "--criterion", "actor_critic_post_edit", "--reset-optimizer", "--use-critic-generator", "--learn-imitate"]
    test_args = ["--use-critic-generator"]
    if use_beam_while_training:
        train_args.append("--use-beam-while-training")
        train_args.extend(["--critic-mix-ratio", str(training_critic_mix_ratio)])
        id += "_beam"
    if alpha is not None:
        train_args.extend(["--alpha", str(alpha)])
        id += f"_alpha_{alpha}"
    
    exp = Experiment(id, i, train_args=train_args, test_args=test_args, task=AC_TASK, base_model_path=base_model_path)
    dict.update(exp.run(try_different_ratio=True))


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]
    run_su_adapt_all(100 + int(sys.argv[1]))
