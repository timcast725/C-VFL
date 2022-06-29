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

results = open('results.txt','w')
results.write(f'Accuracy reached after {100*min_bits_per_epoch} bits sent\n')
results.write('Max Std Best Std\n')

resultsMB = open('resultsMB.txt','w')
resultsMB.write('Avg Std\n')



def all_seeds(prefix, bits):
    print(prefix)
    files = glob.glob(f'./results/C-VFL_{prefix}_seed*_e50,100,200/test_acc5.pkl')

    pickles = []    
    for f in files:
        pkl = pickle.load(open(f, 'rb'))
        #pkl = np.array([x.cpu().numpy() for x in pkl])
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

    final_epoch = 0
    bits_so_far = 0
    while bits_so_far < 10*8*2**20: #10 MB
        bits_so_far += bits_per_epoch
        final_epoch += 1
    ind = np.argmax(avg[0:final_epoch+1])
    max_ind = np.argmax(avg)
    max_error = std[max_ind] 
    best_error = std[ind] 
    results.write(f'&{avg[max_ind]:.2f}\% $\pm$ {max_error:.2f}\%\n')

    bits_to_reach = []
    for i in range(len(pickles)):
        final_epoch = 0
        bits_so_far = 0
        while pickles[i][final_epoch] < 55:
            bits_so_far += bits_per_epoch
            final_epoch += 1
        bits_to_reach.append(bits_so_far/(8*2**30))
    bits_to_reach = np.array(bits_to_reach)
    bits_avg = np.average(bits_to_reach)
    bits_std = np.std(bits_to_reach)

    resultsMB.write(f'& {bits_avg:.2f} $\pm$ {bits_std:.2f}\n')

    return (avg, std, np.array(avg_bits), np.array(std_bits))

#types = ['train_loss', 'train_acc1', 'train_acc5', 'test_loss', 'test_acc1', 'test_acc5']
types = ['test_acc5']
for t in types:
    Qs = [4,8,16]
    for q in Qs:
        bits = math.log(q,2)
        # Parse results
        losses0 = all_seeds(f'b256_le10_lr0.001_comp_quant0_dim1', 32)
        losses1 = all_seeds(f'b256_le10_lr0.001_compquantize_quant{q}_dim1', bits)
        losses2 = all_seeds(f'b256_le10_lr0.001_compquantize_quant{q}_dim2', bits)
        losses3 = all_seeds(f'b256_le10_lr0.001_comptopk_quant{q}_dim1', bits)
    
        fig, ax = plt.subplots()
        # Plot loss
        plt.plot(losses0[0], label='No Compression')
        plt.plot(losses1[0], label='Scalar Quantize')
        plt.plot(losses2[0], label='Vector Quantize')
        plt.plot(losses3[0], label='Top-k')
      
        plt.fill_between(np.linspace(0,100,101), losses0[0] - losses0[1], losses0[0] + losses0[1], alpha=0.3)
        plt.fill_between(np.linspace(0,100,101), losses1[0] - losses1[1], losses1[0] + losses1[1], alpha=0.3)
        plt.fill_between(np.linspace(0,100,101), losses2[0] - losses2[1], losses2[0] + losses2[1], alpha=0.3)
        plt.fill_between(np.linspace(0,100,101), losses3[0] - losses3[1], losses3[0] + losses3[1], alpha=0.3)
    
        plt.xlabel('Epochs')
        if t == 'loss':
            #plt.ylim(0, 1)
            plt.ylabel('Loss')
        else:
            plt.ylabel('Accuracy')
    
        plt.legend(loc='lower right')

        #ratio = 0.5
        #xleft, xright = ax.get_xlim()
        #ybottom, ytop = ax.get_ylim()
        #ax.set_aspect(abs((xright-xleft)/(ybottom-ytop))*ratio)
    
        plt.tight_layout()
        plt.savefig(f'C-VFL{q}.png')
    
        fig, ax = plt.subplots()
        # Plot loss by bits sent 
        min_inds = min(losses0[2].shape[0], losses1[2].shape[0], 
                            losses2[2].shape[0], losses3[2].shape[0])
        x_axis_bits = np.linspace(0, 100*min_bits_per_epoch/(8*2**30), min_inds)

        plt.plot(x_axis_bits, losses0[2][:min_inds], label='No Compression')
        plt.plot(x_axis_bits, losses1[2][:min_inds], label='Scalar Quantize')
        plt.plot(x_axis_bits, losses2[2][:min_inds], label='Vector Quantize')
        plt.plot(x_axis_bits, losses3[2][:min_inds], label='Top-k')
      
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
            #    plt.ylim(0, 90)
            plt.ylabel('Top-5 Accuracy')
    
        plt.legend()
        #ratio = 0.5
        #xleft, xright = ax.get_xlim()
        #ybottom, ytop = ax.get_ylim()
        #ax.set_aspect(abs((xright-xleft)/(ybottom-ytop))*ratio)
    
        plt.tight_layout()
        plt.savefig(f'C-VFL_comm_{q}.png')
        #plt.show()
        #plt.show()
        plt.close()

    
