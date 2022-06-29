import numpy as np
import os
import random

for i in range(5):
    seed = random.randint(0, 2**31)
    os.system(f'python quant_cifar.py 10class/classes/ --num_clients 4 --seed {seed} --b 100 --local_epochs 10 --epochs 200 --lr 0.0001 --quant_level 0') 
    os.system(f'python quant_cifar.py 10class/classes/ --num_clients 4 --seed {seed} --b 100 --local_epochs 10 --epochs 200 --lr 0.0001 --quant_level 4 --vecdim 2 --comp quantize')
    os.system(f'python quant_cifar.py 10class/classes/ --num_clients 4 --seed {seed} --b 100 --local_epochs 10 --epochs 200 --lr 0.0001 --quant_level 4 --vecdim 1 --comp quantize')
    os.system(f'python quant_cifar.py 10class/classes/ --num_clients 4 --seed {seed} --b 100 --local_epochs 10 --epochs 200 --lr 0.0001 --quant_level 4 --vecdim 1 --comp topk')
    os.system(f'python quant_cifar.py 10class/classes/ --num_clients 4 --seed {seed} --b 100 --local_epochs 10 --epochs 200 --lr 0.0001 --quant_level 8 --vecdim 2 --comp quantize')
    os.system(f'python quant_cifar.py 10class/classes/ --num_clients 4 --seed {seed} --b 100 --local_epochs 10 --epochs 200 --lr 0.0001 --quant_level 8 --vecdim 1 --comp quantize')
    os.system(f'python quant_cifar.py 10class/classes/ --num_clients 4 --seed {seed} --b 100 --local_epochs 10 --epochs 200 --lr 0.0001 --quant_level 8 --vecdim 1 --comp topk')
    os.system(f'python quant_cifar.py 10class/classes/ --num_clients 4 --seed {seed} --b 100 --local_epochs 10 --epochs 200 --lr 0.0001 --quant_level 16 --vecdim 2 --comp quantize')
    os.system(f'python quant_cifar.py 10class/classes/ --num_clients 4 --seed {seed} --b 100 --local_epochs 10 --epochs 200 --lr 0.0001 --quant_level 16 --vecdim 1 --comp quantize')
    os.system(f'python quant_cifar.py 10class/classes/ --num_clients 4 --seed {seed} --b 100 --local_epochs 10 --epochs 200 --lr 0.0001 --quant_level 16 --vecdim 1 --comp topk')

