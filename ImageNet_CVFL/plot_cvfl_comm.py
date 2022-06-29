"""
Plot adaptive experimental results
"""
import pickle
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
import os
import glob
import math
import scipy.interpolate as interp

font = {'family' : 'DejaVu Sans',
#        'weight' : 'bold',
        'size'   : 20}
plt.rc('font', **font)
colors=['#6aa2fc', '#fc8181', '#a5ff9e', '#3639ff', '#ff3636', '#13ba00', '#ff62f3']

num_clients = 4
batch_size = 256
embedding_size = 128
server_model_size = embedding_size*num_clients*100 
batches_per_epoch = 495 
min_bits_per_epoch = batches_per_epoch*(num_clients*embedding_size*batch_size*(num_clients+1) + server_model_size*num_clients)*2

def all_seeds(prefix, result_out, num_clients, bits, local_epochs=10):
    print(prefix)
    files = glob.glob(f'./results/C-VFL_{prefix}_seed*_e50,100,200/test_acc5.pkl')
    pickles = []    
    for f in files:
        pkl = pickle.load(open(f, 'rb'))

        if len(pkl) < 101:
            continue
        pkl = pkl[:101]
        pickles.append(np.array(pkl))
    pickles = np.array(pickles)
       
    avg = np.average(pickles, axis=0)
    std = np.std(pickles, axis=0)

    bits_per_epoch = batches_per_epoch*(num_clients*embedding_size*batch_size*(num_clients+1) + server_model_size*num_clients)*bits
    # Make losses by bits sents
    final_epoch = 0
    bits_so_far = 0
    while bits_so_far < 100*min_bits_per_epoch:
        bits_so_far += bits_per_epoch
        final_epoch += 1
    avg_tmp = avg[0:final_epoch+1]
    avg_interp = interp.interp1d(np.arange(avg_tmp.size),avg_tmp)
    avg_bits = avg_interp(np.linspace(0,avg_tmp.size-1, 101))
    std_tmp = std[0:final_epoch+1]
    std_interp = interp.interp1d(np.arange(std_tmp.size),std_tmp)
    std_bits = std_interp(np.linspace(0,std_tmp.size-1, 101))

    return (avg, std, np.array(avg_bits), np.array(std_bits))

types = ['loss', 'accs_train', 'accs_test']
for t in types:
    comps = ['quantize','quantize','topk']
    dims = [1,2,1]
    f = None
    
    for comp,dim in zip(comps,dims):
        # Parse results
        losses0 = all_seeds(f'b256_le10_lr0.001_comp_quant0_dim1',None,4,32)
        losses1 = all_seeds(f'b256_le10_lr0.001_comp{comp}_quant4_dim{dim}',f,4,2)
        losses2 = all_seeds(f'b256_le10_lr0.001_comp{comp}_quant8_dim{dim}',f,4,3)
        losses3 = all_seeds(f'b256_le10_lr0.001_comp{comp}_quant16_dim{dim}',f,4,4)
    
        fig, ax = plt.subplots()
        # Plot loss by bits sent 
        min_inds = min(losses0[2].shape[0], losses1[2].shape[0], 
                            losses2[2].shape[0],losses3[2].shape[0])
        x_axis_bits = np.linspace(0, 100*min_bits_per_epoch/(8*2**30), min_inds)

        plt.plot(x_axis_bits, losses0[2][:min_inds], label='b=32')
        plt.plot(x_axis_bits, losses1[2][:min_inds], label='b=2')
        plt.plot(x_axis_bits, losses2[2][:min_inds], label='b=3')
        plt.plot(x_axis_bits, losses3[2][:min_inds], label='b=4')
      
        plt.fill_between(x_axis_bits, losses0[2][:min_inds] - losses0[3][:min_inds], losses0[2][:min_inds] + losses0[3][:min_inds], alpha=0.3)
        plt.fill_between(x_axis_bits, losses1[2][:min_inds] - losses1[3][:min_inds], losses1[2][:min_inds] + losses1[3][:min_inds], alpha=0.3)
        plt.fill_between(x_axis_bits, losses2[2][:min_inds] - losses2[3][:min_inds], losses2[2][:min_inds] + losses2[3][:min_inds], alpha=0.3)
        plt.fill_between(x_axis_bits, losses3[2][:min_inds] - losses3[3][:min_inds], losses3[2][:min_inds] + losses3[3][:min_inds], alpha=0.3)
    
        plt.xlabel('Communication Cost (GB)')
        if t == 'losses':
            plt.ylim(0.2, 0.6)
            plt.ylabel('Loss')
        else:
            #if t == 'accs_train':
            #    plt.ylim(50, 100)
            #if t == 'accs_test':
            #    plt.ylim(40, 85)
            plt.ylabel('Top-5 Accuracy')
    
        plt.legend()
        #ratio = 0.5
        #xleft, xright = ax.get_xlim()
        #ybottom, ytop = ax.get_ylim()
        #ax.set_aspect(abs((xright-xleft)/(ybottom-ytop))*ratio)
    
        plt.tight_layout()
        plt.savefig(f'C-VFL_comm_{comp}{dim}.png')
        #plt.show()
