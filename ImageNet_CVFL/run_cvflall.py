import numpy as np
import os
import random

quant_levels = [4,8,16]
vec_dims = [1,1,2]
comps = ["topk","quantize","quantize"]

for i in range(5):
    seed = random.randint(0, 2**31)
    os.system(f'python main_cvfl.py -a resnet18 --lr 0.001 --schedule 50 100 --epochs 100 --seed {seed} --quant_level 0 --vecdim 1 --world-size 1 --rank 0 -j 16 --multiprocessing-distributed ./data/imagenet100/')
    for quant_level in quant_levels:
        for vec_dim,comp in zip(vec_dims,comps):
            os.system(f'python main_cvfl.py -a resnet18 --lr 0.001 --schedule 50 100 --epochs 100 --seed {seed} --quant_level {quant_level} --vecdim {vec_dim} --comp {comp} --world-size 1 --rank 0 -j 16 --multiprocessing-distributed ./data/imagenet100/')
