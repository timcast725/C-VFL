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

## C-VFL with ModelNet10 and CIFAR-10

This directory is an extension of the MVCNN-PyTorch for 
experiments with C-VFL: [github.com/RBirkeland/MVCNN-PyTorch](https://github.com/RBirkeland/MVCNN-PyTorch)

### Dataset
Our code uses a 10 class subset of the ModelNet40 dataset.
ModelNet40 12-view PNG dataset can be downloaded from [Google Drive](https://drive.google.com/file/d/0B4v2jR3WsindMUE3N2xiLVpyLW8/view).

Our scripts additionally require that the ModelNet10
dataset is preprocessed and placed in a folder
named '10class/classes/'.

### Dependencies
One can install our environment with Anaconda:
```bash
conda env create -f flearn.yml 
```

### Usage
Example for running C-VFL code with the ModelNet10 dataset with the number of clients=4, number of local iterations Q=10, and vector quantization b=3.
```bash
python quant.py 10class/classes/ --num_clients 4 --b 16 --local_epochs 10 --epochs 50 --lr 0.001 --quant_level 8 --vecdim 2 --comp quantize
```
Example for running C-VFL code with the CIFAR-10 dataset with the number of clients=4, number of local iterations Q=10, and vector quantization b=3.
```bash
python quant_cifar.py 10class/classes/ --num_clients 4 --b 100 --local_epochs 10 --epochs 200 --lr 0.0001 --quant_level 8 --vecdim 2 --comp quantize
```

### Running C-VFL and Plotting Results
If you wish to rerun the experiments from the paper,
the scripts we used to run all experiments sequentially
can be run as follows:
```bash
python run_quant.py
python run_quant_cifar.py
```
This will choose 5 random seeds, run all experiments sequentially,
and places the results in the current working directory.
In our experiments, the seeds that were chosen were:
[707412115,1928644128,16910772,1263880818,1445547577]

Our results are saved as pickle files in the current working directory.
To plot the results:
```bash
python plot_all.py
python plot_comm.py
python plot_12.py
python plot_cifar.py
python plot_comm_cifar.py
```
This will generate all .png plots, as well as the files
'results.txt' and 'resultsMB.txt', which contain the
results seen in Table 2 of the paper.
