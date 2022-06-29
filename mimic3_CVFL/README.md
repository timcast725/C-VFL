# Compressed Vertical Federated Learning

Code for simulating C-VFL, a communication-efficient algorithm for vertically partitioned data.
More details on the algorithm can be in our paper: [**Compressed-VFL: Communication-Efficient Learning with Vertically
Partitioned Data**](https://arxiv.org/abs/2206.08330):

```
@inproceedings{castiglia2022compressed,
  title={Compressed-VFL: Communication-Efficient Learning with Vertically Partitioned Data},
  author={Castiglia, Timothy and Das, Anirban and Wang, Shiqiang and Patterson, Stacy},
  booktitle={International Conference on Machine Learning},
  year={2022}
}
```

## C-VFL on the MIMIC-III dataset

This repo is an extension of the MIMIC-III Benchmarks for C-VFL: [github.com/YerevaNN/mimic3-benchmarks](https://github.com/YerevaNN/mimic3-benchmarks)

### Dependencies
For specific details on downloading and preprocessing the MIMIC-III dataset, 
read the original README below.
Our scripts additionally require that the MIMIC-III
dataset is preprocessed and placed in a folder
named 'data'.

One can install our environment with Anaconda:
```bash
conda env create -f flearn.yml 
```

### Preprocessing the MIMIC-III dataset
First, you will need to get access to the MIMIC-III .csv files: [physionet.org/content/mimiciii-demo/1.4/](https://physionet.org/content/mimiciii-demo/1.4/)
In order to preprocess the data:
    
1. The following command takes MIMIC-III CSVs, generates one directory per `SUBJECT_ID` and writes ICU stay information to `data/{SUBJECT_ID}/stays.csv`, diagnoses to `data/{SUBJECT_ID}/diagnoses.csv`, and events to `data/{SUBJECT_ID}/events.csv`. This step might take around an hour.
```bash
python -m mimic3benchmark.scripts.extract_subjects {PATH TO MIMIC-III CSVs} data/root/
```

2. The following command attempts to fix some issues (ICU stay ID is missing) and removes the events that have missing information. About 80% of events remain after removing all suspicious rows (more information can be found in [`mimic3benchmark/scripts/more_on_validating_events.md`](mimic3benchmark/scripts/more_on_validating_events.md)).

```bash
python -m mimic3benchmark.scripts.validate_events data/root/
```

3. The next command breaks up per-subject data into separate episodes (pertaining to ICU stays). Time series of events are stored in ```{SUBJECT_ID}/episode{#}_timeseries.csv``` (where # counts distinct episodes) while episode-level information (patient age, gender, ethnicity, height, weight) and outcomes (mortality, length of stay, diagnoses) are stores in ```{SUBJECT_ID}/episode{#}.csv```. This script requires two files, one that maps event ITEMIDs to clinical variables and another that defines valid ranges for clinical variables (for detecting outliers, etc.). **Outlier detection is disabled in the current version**.

```bash
python -m mimic3benchmark.scripts.extract_episodes_from_subjects data/root/
```

4. The next command splits the whole dataset into training and testing sets. Note that the train/test split is the same of all tasks.

```bash
python -m mimic3benchmark.scripts.split_train_and_test data/root/
```
	
5. The following command will generate a task-specific dataset for "in-hospital mortality".

```bash
python -m mimic3benchmark.scripts.create_in_hospital_mortality data/root/ data/in-hospital-mortality/
```

After the above commands are done, there will be a directory `data/{task}` for each created benchmark task.
These directories have two sub-directories: `train` and `test`.
Each of them contains bunch of ICU stays and one file with name `listfile.csv`, which lists all samples in that particular set.
Each row of `listfile.csv` has the following form: `icu_stay, period_length, label(s)`.
A row specifies a sample for which the input is the collection of ICU event of `icu_stay` that occurred in the first `period_length` hours of the stay and the target is/are `label(s)`.
In in-hospital mortality prediction task `period_length` is always 48 hours, so it is not listed in corresponding listfiles.

#### Usage
Example for running C-VFL code with the number of clients=4, number of local iterations Q=10, and vector quantization b=3.
```bash
python -um mimic3models.in_hospital_mortality.quant --num_clients 4 --network mimic3models/keras_models/lstm.py --dim 16 --timestep 1.0 --depth 1 --mode train --seed {seed} --lr 0.01 --batch_size 1000 --output_dir mimic3models/in_hospital_mortality --epochs 1000 --local_epochs 10 --quant_level 8 --vecdim 2 --comp quantize
```

### Running experiments from paper
If you wish to rerun the experiments,
the script we used to run all experiments sequentially
can be run as follows:
```bash
python run_quant.py
```
This will choose 5 random seeds, run all experiments sequentially,
and places the results in the current working directory.
In our experiments, the seeds that were chosen were:
[1234420321,1678517510,1679295649,2141512896,755346466].

To speed up loading data at the start of training, pickle
files of the data are saved after the first run of quant.py.
Comment out lines 303-309 and uncomment lines 311-314  
after running once to load from pickle file. 

The results are saved as pickle files in the current working directory.
To plot the results of all experiments using the results in
the quant_all folder, run:
```bash
python plot_all.py
    python plot_comm.py
```
This will generate all .png plots, as well as the files
'results.txt' and 'resultsMB.txt', which contain the
results seen in Table 2 of the paper.

### Citation

If you use this code or these benchmarks in your research, please cite the following publication.
```
@article{Harutyunyan2019,
  author={Harutyunyan, Hrayr and Khachatrian, Hrant and Kale, David C. and Ver Steeg, Greg and Galstyan, Aram},
  title={Multitask learning and benchmarking with clinical time series data},
  journal={Scientific Data},
  year={2019},
  volume={6},
  number={1},
  pages={96},
  issn={2052-4463},
  doi={10.1038/s41597-019-0103-9},
  url={https://doi.org/10.1038/s41597-019-0103-9}
}
```
**Please be sure also to cite the original [MIMIC-III paper](http://www.nature.com/articles/sdata201635).**
The `mimic3benchmark/scripts` directory contains scripts for creating the benchmark datasets.
The reading tools are in `mimic3benchmark/readers.py`.
All evaluation scripts are stored in the `mimic3benchmark/evaluation` directory.
The `mimic3models` directory contains the baselines models along with some helper tools.
Those tools include discretizers, normalizers and functions for computing metrics.

