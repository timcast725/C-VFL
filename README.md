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

### Dependencies
One can install our environment with Anaconda:
```bash
conda env create -f flearn.yml 
```

### Repository Structure
'ModelNet_CVFL': contains code for running C-VFL with the ModelNet10 and CIFAR-10 datasets

'ImageNet_CVFL': contains code for running C-VFL with the ImageNet dataset

'mimic3_CVFL': contains code for running C-VFL with the MIMIC-III dataset

Information on how to run C-VFL are provided in the README's in each folder.
