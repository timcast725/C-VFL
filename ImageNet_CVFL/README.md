# C-VFL with ImageNet dataset

This directory is an extension of the Momentum Contrast (MoCo) with Alignment and Uniformity Losses for C-VFL: [github.com/SsnL/moco_align_uniform](https://github.com/SsnL/moco_align_uniform)

### Dataset
The full ImageNet compatible with PyTorch can be obtained online, e.g., by following instructions specified in [the official PyTorch ImageNet training code](https://github.com/pytorch/examples/tree/master/imagenet#requirements).

#### ImageNet-100 Subset

The ImageNet-100 subset contains a randomly sampled 100 classes of the full ImageNet (1000 classes). The list of the 100 classes we used in our experiments are provided in [`scripts/imagenet100_classes.txt`](./scripts/imagenet100_classes.txt). This subset is identical to the one used in [Contrastive Multiview Coding (CMC)](https://arxiv.org/abs/1906.05849).

We provide a script that constructs proper symlinks to form the subset from the full ImageNet. You may invoke it as following:

```sh
python scripts/create_imagenet_subset.py [PATH_TO_EXISTING_IMAGENET] [PATH_TO_CREATE_SUBSET]
```

Optionally, you may add argument `--subset [PATH_TO_CLASS_SUBSET_FILE]` to specify a custom subset file, which should follow the same format as [`scripts/imagenet100_classes.txt`](./scripts/imagenet100_classes.txt). See [`scripts/create_imagenet_subset.py`](./scripts/create_imagenet_subset.py) for more options.

### Running C-VFL code

#### Dependencies
One can install our environment with Anaconda:
```bash
conda env create -f flearn.yml 
```

#### Usage
```
usage: main_cvfl.py [-h] [-a ARCH] [-j N] [--epochs N] [--start-epoch N]
                    [-b N] [--lr LR] [--lr_un LR]
                    [--schedule [SCHEDULE [SCHEDULE ...]]] [--wd W] [-p N]
                    [--resume PATH] [-e] [--world-size WORLD_SIZE]
                    [--rank RANK] [--dist-url DIST_URL]
                    [--dist-backend DIST_BACKEND] [--seed SEED]
                    [--gpus GPUS [GPUS ...]] [--multiprocessing-distributed]
                    [--num_clients NUM_CLIENTS] [--quant_level QUANT_LEVEL]
                    [--vecdim VECDIM] [--comp COMP]
                    DIR
```

#### Running experiments from paper
If you wish to rerun the experiments,
the script we used to run all experiments sequentially
can be run as follows:
```bash
python run_cvflall.py
```
Our scripts expect ImageNet100 to be placed in the folder "./data/imagenet100/".
This will choose 5 random seeds, run all experiments sequentially,
and places the results in the current working directory.
In our experiments, the seeds that were chosen were:
[707412115,1928644128,16910772,1263880818,1445547577]

The results are saved as pickle files in the 'results' folder.
To plot the results:
```bash
python plot_cvfl.py
python plot_cvfl_comm.py
```
This will generate all plots as .png files.

## License

This project is under the CC-BY-NC 4.0 license. See [LICENSE](LICENSE) for details.
