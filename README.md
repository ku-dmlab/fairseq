
### Executing the experiments

First run the script to download and preprocess the data:
```bash
bash preprocess.sh
bash preprocess_wmt.sh
```

The offline RL dataset should be generated to run the code
```
python build_offline_BLEU_dataset.py <seed>
python build_offline_mss_dataset.py <seed>
```

Then, imply run the experiment by running
```
python run_experiment_post_edit <seed>
```



