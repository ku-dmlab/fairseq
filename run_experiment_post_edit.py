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
        if base_model_path is None:
            self.train_args += []#["--patience", "3"]
        else:
            self.train_args += ["--validate-interval-updates", "4000"]#, "--patience", "80"]
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
            for ratio in [5.0, 2.0, 1.0, 0.5]:#[1000.0, 5.0, 2.0, 1.0, 0.5, 0.25, 0.01]:
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
    #run_baseline_wmt(i, all_scores)
    #run_supervised_adaptation(i, all_scores)
    # alpha search
    # tau search
    # reward scale search
    
    def save_scores(ind):
        res_file = os.path.join(RESULTS_DIR, f"result_cr1_{i}_{ind}.pkl")
        with open(res_file, "wb") as f:
            pickle.dump(all_scores, f)

    #run_offline(i, all_scores, model="baseline_wmt", offline_data="su_adapt", alpha=1.0, tau=0.99, critic_mix_ratio=1000)
    #run_offline(i, all_scores, model="su_adapt", offline_data="su_adapt", alpha=1.0, tau=0.99, critic_mix_ratio=1000, test_only=True, force_id="offline_real_final_1_baseline_wmt_su_adapt_alpha_1.0_tau_0.99_normalize_advantage_reward-scaler_100")

    #run_offline(i, all_scores, model="baseline", alpha=1.0, tau=0.99, critic_mix_ratio=1000, adapt=False, do_not_restore=True)
    #run_offline(i, all_scores, model="baseline", alpha=1.0, tau=0.99, critic_mix_ratio=1000, adapt=False, test_only=True, force_id="offline_real_final_baseline_adapt_False_alpha_1.0_tau_0.99")
    
    #run_offline(i, all_scores, model="baseline_wmt", offline_data="su_adapt", critic_mix_ratio=1000, use_pcl=True, test_only=True)
    #run_offline(i, all_scores, model="su_adapt", offline_data="su_adapt", alpha=1.0, tau=0.99, critic_mix_ratio=1000, test_only=True, use_pcl=True, force_id="offline_alpha_baseline_wmt_adapt_Truesu_adapt_pcl")
    
    #run_offline(i, all_scores, model="baseline", offline_data="su_adapt", critic_mix_ratio=1000, use_pcl=True, adapt=False, do_not_restore=True)
    #run_offline(i, all_scores, model="baseline", alpha=1.0, tau=0.99, critic_mix_ratio=1000, adapt=False, test_only=True, use_pcl=True, force_id="offline_alpha_baseline_adapt_False_pcl")

    #run_offline(i, all_scores, model="baseline_wmt", offline_data="su_adapt", alpha=1.0, tau=0.99, critic_mix_ratio=1000, use_full=True)
    #run_offline(i, all_scores, model="su_adapt", offline_data="su_adapt", alpha=1.0, tau=0.99, critic_mix_ratio=1000, test_only=True, use_full=True, force_id="offline_alpha_baseline_wmt_adapt_Truesu_adapt_alpha_1.0_tau_0.99_full")

    #run_offline(i, all_scores, model="baseline", alpha=1.0, tau=0.99, critic_mix_ratio=1000, adapt=False, do_not_restore=True, use_full=True)
    #run_offline(i, all_scores, model="baseline", alpha=1.0, tau=0.99, critic_mix_ratio=1000, adapt=False, test_only=True, use_full=True, force_id="offline_alpha_baseline_adapt_False_alpha_1.0_tau_0.99_full")
    
    #run_reinforce(i, all_scores, model="baseline", adapt=False, test_only=True)
    #run_reinforce_test(i, all_scores, model="baseline", adapt=False, force_id="reinforce")
    #run_reinforce(i, all_scores, model="su_adapt", adapt=True)
    #run_reinforce_test(i, all_scores, model="su_adapt", adapt=True)
    
    #run_baseline_with_offline(i, all_scores, test_only=True)
    #run_baseline(i, all_scores, test_only=True)
    #su_adapt_with_offline(i, all_scores, test_only=True)
    #run_supervised_adaptation(i, all_scores, test_only=True)
    #run_baseline_wmt(i, all_scores, test_only=True, data_path="data-bin/wmt17.en-de.iwslt")
    
    # alpha = {100:0.0, 101:0.1, 102:1.0, 103:10.0}
    # alpha2 = {100:0.3, 101:3.0, 102:30.0, 103:100.0}
    # save_scores(-1)
    # run_offline(100, all_scores, model="baseline", adapt=False, alpha=alpha[i], tau=0.99, critic_mix_ratio=1000)
    # print(all_scores)
    # save_scores(0)
    # run_offline(100, all_scores, model="baseline", adapt=False, alpha=alpha2[i], tau=0.99, critic_mix_ratio=1000)
    # print(all_scores)
    # save_scores(1)
    # tau = {100:0.9, 101:0.95, 102:0.97, 103:0.999}
    # tau2 = {100:0.5, 101:0.6, 102:0.7, 103:0.8}
    # run_offline(100, all_scores, model="baseline", adapt=False, alpha=1.0, tau=tau[i], critic_mix_ratio=1000)
    # print(all_scores)
    # save_scores(2)
    # run_offline(100, all_scores, model="baseline", adapt=False, alpha=1.0, tau=tau2[i], critic_mix_ratio=1000)
    # print(all_scores)
    # save_scores(3)
    # run_baseline(i, all_scores)
    # run_baseline_continued(i, all_scores)
    #run_reinforce_is(i, all_scores, model="baseline_80", adapt=False, max_epoch=320)
    #run_reinforce_is(i, all_scores, model="baseline_80", adapt=False, max_epoch=320, use_full=True)
    #run_reinforce_is(i, all_scores, model="baseline_80", adapt=False, use_reinforce=True)
    #run_reinforce_is(i, all_scores, model="baseline_80", adapt=False, use_reinforce=True, use_bc=True, max_epoch=320)
    #run_baseline_wmt(i, all_scores)
    #run_offline_only(i, all_scores, model="baseline_40", max_epoch=320, use_reinforce=True)
    #run_offline_only(i, all_scores, model="baseline_40", max_epoch=320, alpha=1.0, tau=0.999, use_gold_reward=True, test_only=True)
    
    #run_offline_only(i, all_scores, model="baseline_wmt", max_epoch=320, alpha=1.0, tau=0.999, adapt=True, use_reinforce=True, test_only=True)
    # 13.79, 14.46, 14.28, 13.93 -> 14.115
    #run_offline_only(i, all_scores, model="baseline_wmt", max_epoch=320, alpha=1.0, tau=0.999, adapt=True, test_only=True)
    # 13.86, 14.12, 13.82, 13.63
    #run_offline_only(i, all_scores, model="baseline_wmt", max_epoch=320, alpha=100, tau=0.999, adapt=True, use_reward_weighted=True, test_only=True)
    # 14.19, 14.28, 14.01, 13.85 -> 14.0825
    rs = {100:30, 101:10, 102:3, 103:1}
    #run_offline_only(100, all_scores, model="baseline_wmt", max_epoch=320, alpha=1.0, tau=0.999, adapt=True, use_pcl=True, reward_scaler=rs[i])

    #run_offline_only(i, all_scores, model="baseline_wmt", max_epoch=45, alpha=1.0, tau=0.999, adapt=True, use_reinforce=True, use_bc=True, num=6)
    #run_offline_only(i, all_scores, model="baseline_wmt", max_epoch=45, alpha=1.0, tau=0.999, adapt=True, use_reinforce=True, test_only=True, num=6)
    #run_offline_only(i, all_scores, model="baseline_wmt", max_epoch=45, alpha=1.0, tau=0.999, adapt=True, test_only=True, num=6)
    #run_offline_only(i, all_scores, model="baseline_wmt", max_epoch=45, alpha=100, tau=0.999, adapt=True, use_reward_weighted=True, test_only=True, num=6)
    #run_offline_only(i, all_scores, model="baseline_wmt", max_epoch=45, alpha=1.0, tau=0.999, adapt=True, use_pcl=True, reward_scaler=1)

    #run_baseline_half(i, all_scores, max_epoch=40, portion=0.5) #translation_portion
    #run_baseline(i, all_scores, max_epoch=100)

    #for num in [12, 13, 14, 15]:
        #run_offline_only(i, all_scores, model="baseline_40_0.5", max_epoch=50, use_reinforce=True, use_bc=True, num=num, test_only=True, no_ensemble=True)
        #run_offline_only(i, all_scores, model="baseline_40_0.5", max_epoch=50, alpha=100, tau=0.999, num=num)
        #run_offline_only(i, all_scores, model="baseline_40_0.5", max_epoch=50, use_reinforce=True, num=num, test_only=True, no_ensemble=True)
        #run_offline_only(i, all_scores, model="baseline_40_0.5", max_epoch=50, alpha=100, tau=0.999, use_reward_weighted=True, num=num)
    run_offline_only(i, all_scores, model="baseline_40_0.5", max_epoch=50, use_reinforce=True, num=16, no_ensemble=True, test_only=True)
    #run_offline_only(i, all_scores, model="baseline_40_0.5", max_epoch=50, alpha=100, tau=0.999, use_reward_weighted=True, num=16)
    run_offline_only(i, all_scores, model="baseline_40_0.5", max_epoch=50, use_reinforce=True, use_bc=True, num=16, no_ensemble=True, test_only=True)
    #run_offline_only(i, all_scores, model="baseline_40_0.5", max_epoch=50, alpha=100, tau=0.999, num=16)
    run_offline_only(i, all_scores, model="baseline_wmt", max_epoch=50, use_reinforce=True, num=16, adapt=True, no_ensemble=True, test_only=True)
    # run_offline_only(i, all_scores, model="baseline_wmt", max_epoch=50, alpha=100, tau=0.999, use_reward_weighted=True, num=16, adapt=True)
    run_offline_only(i, all_scores, model="baseline_wmt", max_epoch=50, use_reinforce=True, use_bc=True, num=16, adapt=True, no_ensemble=True, test_only=True)
    # run_offline_only(i, all_scores, model="baseline_wmt", max_epoch=50, alpha=100, tau=0.999, num=16, adapt=True)

    #run_baseline_wmt(i, all_scores, test_only=True, data_path="data-bin/wmt17.en-de.iwslt")
    #for num in [6, 7, 8, 9]:
    #    run_offline_only(i, all_scores, model="baseline_wmt", max_epoch=45, alpha=1.0, tau=0.999, adapt=True, use_reinforce=True, use_bc=True, num=num, test_only=True, no_ensemble=True)
    #    run_offline_only(i, all_scores, model="baseline_wmt", max_epoch=45, alpha=1.0, tau=0.999, adapt=True, use_reinforce=True, test_only=True, num=num, no_ensemble=True)

    # run_baseline(i, all_scores, max_epoch=40)
    # run_baseline_continued(i, all_scores, max_epoch=80, continue_from=40)
    # run_reinforce_is(i, all_scores, model="baseline_40", adapt=False, max_epoch=80)
    # run_reinforce_is(i, all_scores, model="baseline_40", adapt=False, use_reinforce=True, max_epoch=80)
    # run_reinforce_is(i, all_scores, model="baseline_40", adapt=False, use_reinforce=True, use_bc=True, max_epoch=80)

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

def run_baseline_half(i, dict, test_only=False, max_epoch=80, portion=0.5):
    TASK = "translation_portion"
    id = f"baseline_{max_epoch}_{portion}"
    train_args = ["--lr", "5e-4", "--criterion", "label_smoothed_cross_entropy",
        "--label-smoothing", "0.1", "--max-epoch", str(max_epoch), "--portion", str(portion)]
    exp = Experiment(id, i, train_args=train_args, test_only=test_only, task=TASK)
    dict.update(exp.run())

def run_baseline(i, dict, test_only=False, max_epoch=80):
    id = f"baseline_{max_epoch}"
    train_args = ["--lr", "5e-4", "--criterion", "label_smoothed_cross_entropy",
        "--label-smoothing", "0.1", "--max-epoch", str(max_epoch)]
    exp = Experiment(id, i, train_args=train_args, test_only=test_only)
    dict.update(exp.run())

def run_baseline_continued(i, dict, test_only=False, max_epoch=160, continue_from=80):
    base_model_path = os.path.join(BASE_DIR, f"baseline_{continue_from}", str(i), "checkpoint_best.pt")
    id = f"baseline_cont_{max_epoch}"
    train_args = ["--lr", "5e-4", "--criterion", "label_smoothed_cross_entropy",
        "--label-smoothing", "0.1", "--max-epoch", str(max_epoch), "--restore-file", base_model_path]
    exp = Experiment(id, i, train_args=train_args, test_only=test_only)
    dict.update(exp.run())

def run_baseline_wmt(i, dict, test_only=False, data_path="data-bin/wmt17.en-de"):
    id = "baseline_wmt"
    train_args = ["--lr", "5e-4", "--criterion", "label_smoothed_cross_entropy",
        "--label-smoothing", "0.1", "--max-epoch", "40"]
    exp = Experiment(id, i, train_args=train_args, data_path=data_path, test_only=test_only)
    dict.update(exp.run())

def run_baseline_with_offline(i, dict, test_only=False):
    id = "baseline_with_offline"
    train_args = ["--lr", "5e-4", "--criterion", "label_smoothed_cross_entropy",
        "--label-smoothing", "0.1", "--max-epoch", "80"]
    data_path = f"data-bin/iwslt14.tokenized.en-de:data-bin/iwslt14.tokenized.offline.{i}.en-de"
    exp = Experiment(id, i, train_args=train_args, data_path=data_path, test_only=test_only)
    dict.update(exp.run())

def su_adapt_with_offline(i, dict, test_only=False):
    id = "baseline_wmt_with_offline"
    train_args = ["--lr", "5e-4", "--criterion", "label_smoothed_cross_entropy",
        "--label-smoothing", "0.1", "--max-epoch", "80"]
    data_path = f"data-bin/wmt17.en-de.iwslt:data-bin/wmt17.iwslt.offline_su_adapt.{i}.en-de"
    exp = Experiment(id, i, train_args=train_args, data_path=data_path, test_only=test_only)
    dict.update(exp.run())


def run_supervised_adaptation(i, dict, use_offline=False, test_only=True):
    base_model_path = os.path.join(BASE_DIR, "baseline_wmt", str(i), "checkpoint_best.pt")
    id = "su_adapt"
    train_args = ["--lr", "5e-4", "--criterion", "label_smoothed_cross_entropy",
        "--label-smoothing", "0.1"]
    data_path = "data-bin/wmt17.en-de.iwslt"
    if use_offline:
        data_path = f"data-bin/wmt17.en-de.iwslt:data-bin/wmt17.iwslt.offline_su_adapt.{i}.en-de"
        id += "_wmt"
    exp = Experiment(id, i, train_args=train_args, data_path=data_path, base_model_path=base_model_path, test_only=test_only)
    dict.update(exp.run())

def run_offline(i, dict, model="baseline_wmt", offline_data="baseline_wmt", alpha=None, tau=None, critic_mix_ratio=1000, use_pcl=False, 
            use_iql=False, test_only=False, adapt=True, do_not_restore=False, use_full=False, force_id=None, subtract_max=False):
    base_model_path = os.path.join(BASE_DIR, model, str(i), "checkpoint_best.pt")
    TASK = "translation_with_actor_critic_offline"

    # datasets = []
    # for beam_ind in range(beam_ind_end):
    #     datasets.append(f"data-bin/wmt17.iwslt.offline_{model}_best_{beam_ind}.{i}.en-de")
    # datasets = ",".join(datasets)
    datasets = f"data-bin/wmt17.iwslt.offline_{offline_data}.{i}.en-de" if adapt else f"data-bin/iwslt14.tokenized.offline.{i}.en-de"
    
    id = f"offline_hyp_alpha_{model}_adapt_{adapt}"
    if adapt:
        id += offline_data

    train_args = ["--lr", "5e-4", "--criterion", "actor_critic_offline", "--reset-optimizer", "--use-critic-generator",
                  "--offline-data", datasets, "--max-epoch", "63", "--use-beam-while-training", "--critic-mix-ratio", str(critic_mix_ratio)]
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
    if use_full:
        train_args.append("--use-full")
        id += f"_full"

    if force_id is not None:
        id = force_id
    if subtract_max:
        test_args.append("--subtract-max")

    data_path = "data-bin/wmt17.en-de.iwslt" if adapt else"data-bin/iwslt14.tokenized.en-de"
    exp = Experiment(id, i, train_args=train_args, test_args=test_args, task=TASK, base_model_path=base_model_path,
                     data_path=data_path, test_only=test_only, do_not_restore=do_not_restore)
    dict.update(exp.run(try_different_ratio=True))

def run_offline_only(i, dict, model="baseline", test_only=False, adapt=False, use_reinforce=False, use_bc=False, max_epoch=160, use_full=False, alpha=None, tau=None, use_gold_reward=False, use_reward_weighted=False, use_pcl=False, reward_scaler=None, num=9, no_ensemble=False):
    base_model_path = os.path.join(BASE_DIR, model, str(i), "checkpoint_last.pt")
    TASK = "translation_with_actor_critic_offline"
    noises = {9:5, 8:3, 7:1, 6:0, 10:10, 11:15, 12:"0_1", 13:"0_2", 14:"0_3", 15:"0_4"}
    if num == 16:
        datasets = f"data-bin/wmt17.iwslt.mss_baseline_wmt_best_5_50_0_0.{i}.en-de" if adapt else f"data-bin/iwslt.mss_baseline_40_0.5_best_5_50_0_0.{i}.en-de"
    else:
        datasets = f"data-bin/wmt17.iwslt.good_baseline_wmt_best_5_50_{noises[num]}.{i}.en-de" if adapt else f"data-bin/iwslt.good_baseline_40_0.5_best_5_50_{noises[num]}.{i}.en-de"
   
    id = f"offline_only_{num}_{model}_{max_epoch}_use_bc_{use_bc}_use_reinforce_{use_reinforce}_adapt_{adapt}"

    train_args = ["--lr", "1e-4", "--criterion", "actor_critic_offline", "--reset-optimizer", "--use-critic-generator",
                  "--offline-data", datasets, "--max-epoch", str(max_epoch), "--offline-only", "--use-beam-while-training", "--critic-mix-ratio", str(1.0)]
    if no_ensemble:
        test_args = []
    else:
        test_args = ["--use-critic-generator"]

    if alpha is not None:
        train_args.extend(["--alpha", str(alpha)])
        id += f"_alpha_{alpha}"
    if tau is not None:
        train_args.extend(["--tau", str(tau)])
        id += f"_tau_{tau}"
    
    if use_pcl:
        train_args.append("--use-pcl")
        id += f"_pcl"
    if use_reinforce:
        train_args.append("--use-reinforce")
    if use_bc:
        train_args.append("--use-bc")
    if use_full:
        train_args.append("--use-full")
        id += f"_full"
    if use_gold_reward:
        train_args.append("--use-gold-reward")
        id += f"_gold"
    if use_reward_weighted:
        train_args.append("--use-reward-weighted")
        id += f"_reward_weighted"
    if reward_scaler is not None:
        train_args.extend(["--reward-scaler", str(reward_scaler)])
        id += f"_rs_{reward_scaler}"

    data_path = "data-bin/wmt17.en-de.iwslt" if adapt else"data-bin/iwslt14.tokenized.en-de"
    exp = Experiment(id, i, train_args=train_args, test_args=test_args, base_model_path=base_model_path, task=TASK, data_path=data_path, test_only=test_only)
    dict.update(exp.run(try_different_ratio=not no_ensemble))

def run_reinforce_is(i, dict, model="baseline", offline_data="baseline_wmt", test_only=False, adapt=False, use_reinforce=False, use_bc=False, max_epoch=160, use_full=False):
    base_model_path = os.path.join(BASE_DIR, model, str(i), "checkpoint_last.pt")
    TASK = "translation_with_actor_critic_offline"
    datasets = f"data-bin/wmt17.iwslt.offline_{offline_data}.{i}.en-de" if adapt else f"data-bin/iwslt14.tokenized.offline.{i}.en-de"
    
    id = f"reinforce_is_{model}_{max_epoch}_adapt_{adapt}_use_bc_{use_bc}_use_reinforce_{use_reinforce}"
    if adapt:
        id += offline_data

    train_args = ["--lr", "5e-4", "--criterion", "actor_critic_offline", "--reset-optimizer",
                  "--offline-data", datasets, "--max-epoch", str(max_epoch)]
    if use_reinforce:
        train_args.append("--use-reinforce")
    if use_bc:
        train_args.append("--use-bc")
    if use_full:
        train_args.append("--use-full")
        id += f"_full"
    data_path = "data-bin/wmt17.en-de.iwslt" if adapt else"data-bin/iwslt14.tokenized.en-de"
    exp = Experiment(id, i, train_args=train_args, base_model_path=base_model_path, task=TASK, data_path=data_path, test_only=test_only)
    dict.update(exp.run())

def run_reinforce(i, dict, model="baseline", test_only=False, adapt=False):
    base_model_path = os.path.join(BASE_DIR, model, str(i), "checkpoint_last.pt")
    
    id = f"reinforce_{model}_adapt_{adapt}"
    train_args = ["--lr", "5e-5", "--criterion", "policy_gradient", "--reset-optimizer", "--max-update", "25000", "--fp16"]
    if adapt:
        train_args.extend(["--sample-beam", "3"])
    data_path = "data-bin/wmt17.en-de.iwslt" if adapt else"data-bin/iwslt14.tokenized.en-de"
    exp = Experiment(id, i, train_args=train_args, base_model_path=base_model_path, data_path=data_path, test_only=test_only, max_tokens=2048)
    dict.update(exp.run())

def run_reinforce_test(i, dict, model="baseline", adapt=False, force_id=None):
    base_model_path = os.path.join(BASE_DIR, model, str(i), "checkpoint_last.pt")
    
    id = f"reinforce_{model}_adapt_{adapt}"
    train_args = ["--lr", "5e-5", "--criterion", "policy_gradient", "--reset-optimizer", "--max-update", "25000", "--fp16"]
    if adapt:
        train_args.extend(["--sample-beam", "3"])
    data_path = "data-bin/wmt17.en-de.iwslt" if adapt else"data-bin/iwslt14.tokenized.en-de"
    if force_id is not None:
        id = force_id
    test_args = ["--use-critic-generator"]

    exp = Experiment(id, i, train_args=train_args, test_args=test_args, task="translation_with_actor_critic", base_model_path=base_model_path, data_path=data_path, test_only=True, max_tokens=2048)
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
    sys.argv.append("1")
    os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]
    run_su_adapt_all(100 + int(sys.argv[1]))
