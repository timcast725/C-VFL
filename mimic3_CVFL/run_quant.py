import numpy as np
import random 
import os

for i in range(5):
    seed = random.randint(0,2**31)
    os.system(f'python -um mimic3models.in_hospital_mortality.quant --num_clients 4 --network mimic3models/keras_models/lstm.py --dim 16 --timestep 1.0 --depth 1 --mode train --seed {seed} --lr 0.01 --batch_size 1000 --output_dir mimic3models/in_hospital_mortality --epochs 1000 --local_epochs 10 --quant_level 0')
    os.system(f'python -um mimic3models.in_hospital_mortality.quant --num_clients 4 --network mimic3models/keras_models/lstm.py --dim 16 --timestep 1.0 --depth 1 --mode train --seed {seed} --lr 0.01 --batch_size 1000 --output_dir mimic3models/in_hospital_mortality --epochs 1000 --local_epochs 10 --quant_level 4 --vecdim 2 --comp quantize')
    os.system(f'python -um mimic3models.in_hospital_mortality.quant --num_clients 4 --network mimic3models/keras_models/lstm.py --dim 16 --timestep 1.0 --depth 1 --mode train --seed {seed} --lr 0.01 --batch_size 1000 --output_dir mimic3models/in_hospital_mortality --epochs 1000 --local_epochs 10 --quant_level 8 --vecdim 2 --comp quantize')
    os.system(f'python -um mimic3models.in_hospital_mortality.quant --num_clients 4 --network mimic3models/keras_models/lstm.py --dim 16 --timestep 1.0 --depth 1 --mode train --seed {seed} --lr 0.01 --batch_size 1000 --output_dir mimic3models/in_hospital_mortality --epochs 1000 --local_epochs 10 --quant_level 16 --vecdim 2 --comp quantize')
    os.system(f'python -um mimic3models.in_hospital_mortality.quant --num_clients 4 --network mimic3models/keras_models/lstm.py --dim 16 --timestep 1.0 --depth 1 --mode train --seed {seed} --lr 0.01 --batch_size 1000 --output_dir mimic3models/in_hospital_mortality --epochs 1000 --local_epochs 10 --quant_level 4 --comp quantize')
    os.system(f'python -um mimic3models.in_hospital_mortality.quant --num_clients 4 --network mimic3models/keras_models/lstm.py --dim 16 --timestep 1.0 --depth 1 --mode train --seed {seed} --lr 0.01 --batch_size 1000 --output_dir mimic3models/in_hospital_mortality --epochs 1000 --local_epochs 10 --quant_level 8 --comp quantize')
    os.system(f'python -um mimic3models.in_hospital_mortality.quant --num_clients 4 --network mimic3models/keras_models/lstm.py --dim 16 --timestep 1.0 --depth 1 --mode train --seed {seed} --lr 0.01 --batch_size 1000 --output_dir mimic3models/in_hospital_mortality --epochs 1000 --local_epochs 10 --quant_level 16 --comp quantize')
    os.system(f'python -um mimic3models.in_hospital_mortality.quant --num_clients 4 --network mimic3models/keras_models/lstm.py --dim 16 --timestep 1.0 --depth 1 --mode train --seed {seed} --lr 0.01 --batch_size 1000 --output_dir mimic3models/in_hospital_mortality --epochs 1000 --local_epochs 10 --comp topk --quant_level 4')
    os.system(f'python -um mimic3models.in_hospital_mortality.quant --num_clients 4 --network mimic3models/keras_models/lstm.py --dim 16 --timestep 1.0 --depth 1 --mode train --seed {seed} --lr 0.01 --batch_size 1000 --output_dir mimic3models/in_hospital_mortality --epochs 1000 --local_epochs 10 --comp topk --quant_level 8')
    os.system(f'python -um mimic3models.in_hospital_mortality.quant --num_clients 4 --network mimic3models/keras_models/lstm.py --dim 16 --timestep 1.0 --depth 1 --mode train --seed {seed} --lr 0.01 --batch_size 1000 --output_dir mimic3models/in_hospital_mortality --epochs 1000 --local_epochs 10 --comp topk --quant_level 16')
    os.system(f'python -um mimic3models.in_hospital_mortality.quant --num_clients 4 --network mimic3models/keras_models/lstm.py --dim 16 --timestep 1.0 --depth 1 --mode train --seed {seed} --lr 0.01 --batch_size 1000 --output_dir mimic3models/in_hospital_mortality --epochs 10000 --local_epochs 1 --quant_level 8 --vecdim 2 --comp quantize')
    os.system(f'python -um mimic3models.in_hospital_mortality.quant --num_clients 4 --network mimic3models/keras_models/lstm.py --dim 16 --timestep 1.0 --depth 1 --mode train --seed {seed} --lr 0.01 --batch_size 1000 --output_dir mimic3models/in_hospital_mortality --epochs 1000 --local_epochs 25 --quant_level 8 --vecdim 2 --comp quantize')
    os.system(f'python -um mimic3models.in_hospital_mortality.quant --num_clients 12 --network mimic3models/keras_models/lstm.py --dim 16 --timestep 1.0 --depth 1 --mode train --seed {seed} --lr 0.01 --batch_size 1000 --output_dir mimic3models/in_hospital_mortality --epochs 1000 --local_epochs 10 --quant_level 8 --vecdim 2 --comp quantize')
