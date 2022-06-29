"""
Compressed VFL training with the MIMIC-III dataset
"""

import numpy as np
import argparse
import os
import imp
import re
import math
import copy
import random
import itertools
import torch

from sklearn.cluster import KMeans
from mimic3models.in_hospital_mortality import utils
from mimic3benchmark.readers import InHospitalMortalityReader
from mimic3models.preprocessing import Discretizer, Normalizer
from mimic3models import metrics
from mimic3models import keras_utils
from mimic3models import common_utils
import tensorflow as tf
from keras.callbacks import ModelCheckpoint, CSVLogger
from tensorflow.python.eager import backprop
import pickle
from sklearn import metrics as skmetrics
from tqdm import tqdm
from tensorflow.keras.models import Model

from scipy.optimize import minimize
from scipy.optimize import Bounds
from scipy.optimize import NonlinearConstraint
from scipy.optimize import BFGS

import latbin

def sparsify(tensor, compress_ratio):
    elemnum = tensor.get_shape().as_list()[0]
    k = max(1, int(elemnum * compress_ratio))
    _, indices = tf.math.top_k(tf.math.abs(tensor), k, sorted=False)
    values = tf.gather(tensor, indices)
    return values, indices


def desparsify(indices, values, tensor_size):
    indices = tf.expand_dims(indices, 1)
    tensor = tf.scatter_nd(indices, values, [tensor_size])
    return tensor


class TopKCompressor():
    def __init__(self, compress_ratio):
        self.compress_ratio = compress_ratio

    def compress(self, tensor):
        tensor_shape = tf.shape(tensor)
        tensor_flatten = tf.reshape(tensor, [-1])
        elemnum = tensor_flatten.get_shape().as_list()[0]

        values, indices = sparsify(tensor_flatten, self.compress_ratio)

        indices = tf.cast(indices, tf.int32)
        values = tf.bitcast(values, tf.int32)
        tensor_compressed = tf.concat([values, indices], 0)
        ctx = tensor_shape, elemnum
        return [tensor_compressed], ctx

    def decompress(self, tensors_compressed, ctx):
        """Decompress by filling empty slots with zeros and reshape back using the original shape"""
        tensor_compressed, = tensors_compressed
        values, indices = tf.split(tensor_compressed, 2)
        values = tf.bitcast(values, tf.float32)
        tensor_shape, tensor_size = ctx
        tensor_decompressed = desparsify(indices, values, tensor_size)

        tensor_decompressed = tf.reshape(tensor_decompressed, tensor_shape)
        return tensor_decompressed

def argparser():
    """
    Parse input arguments
    """
    import sys
    workers = int(sys.argv[2])
    parser = argparse.ArgumentParser()
    common_utils.add_common_arguments(parser)
    parser.add_argument('--seed', type=int, nargs='?', default=42,
                            help='Random seed to be used.')
    parser.add_argument('--target_repl_coef', type=float, default=0.0)
    parser.add_argument('--data', type=str, help='Path to the data of in-hospital mortality task',
                        default=os.path.join(os.path.dirname(__file__), '../../data/in-hospital-mortality/'))
    parser.add_argument('--output_dir', type=str, help='Directory relative which all output files are stored',
                        default='.')
    parser.add_argument('--num_clients', type=int, help='Number of clients to split data between vertically',
                        default=2)
    parser.add_argument('--local_epochs', type=int, help='Number of local epochs to run at each client before synchronizing',
                        default=1)
    parser.add_argument('--quant_level', type=int, help='Level of quantization on embeddings',
                        default=0)
    parser.add_argument('--correct', type=bool, help='Add error correction to algorithm',
                        default=False)
    parser.add_argument('--vecdim', type=int, help='Vector quantization dimension',
                        default=1)
    parser.add_argument('--comp', type=str, help='Which compressor', default="")
                        
    args = parser.parse_args()
    print("*"*80, "\n\n", args, "\n\n", "*"*80)
    return args

if __name__ == "__main__":
    # Parse input arguments
    args = argparser()
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    random.seed(args.seed)
    num_clients = args.num_clients
    local_epochs = args.local_epochs
    lr = args.lr

    def quantize_vector(x, quant_min=-1, quant_max=1, quant_level=5, dim=2):
        """Uniform vector quantization approach

        Args:
            x: Original signal
            quant_min: Minimum quantization level
            quant_max: Maximum quantization level
            quant_level: Number of quantization levels
            dim: dimension of vectors to quantize

        Returns:
            x_quant: Quantized signal

        Currently only works for 2 dimensions and 
        quant_levels of 4, 8, and 16.
        """

        dither = np.random.uniform(-(quant_max-quant_min)/(2*(quant_level-1)), 
                                    (quant_max-quant_min)/(2*(quant_level-1)),
                                    size=np.array(x).shape) 
        x_normalize = np.array(x) + dither

        # Move into 0,1 range:
        x_normalize = x_normalize/2 + 1

        A2 = latbin.lattice.ALattice(dim,scale=1/(2*math.log(quant_level,2)))
        if quant_level == 4:
            A2 = latbin.lattice.ALattice(dim,scale=1/4)
        elif quant_level == 8:
            A2 = latbin.lattice.ALattice(dim,scale=1/8.5)
        elif quant_level == 16:
            A2 = latbin.lattice.ALattice(dim,scale=1/19)
        #if dim == 4:
        #    A2 = latbin.lattice.DLattice(dim,scale=1/(4*math.log(quant_level,2)))
        #elif dim == 8:
        #    A2 = latbin.lattice.ELattice(dim,scale=1/(4*math.log(quant_level,2)))
        
        for i in range(0, x_normalize.shape[1], dim):
            x_normalize[:,i:(i+dim)] = A2.lattice_to_data_space(
                                            A2.quantize(x_normalize[:,i:(i+dim)]))

        # Move out of 0,1 range:
        x_normalize = 2*(x_normalize - 1)
        return tf.convert_to_tensor(x_normalize - dither, dtype=tf.float32)

    def quantize_scalar(x, quant_min=-1, quant_max=1, quant_level=5):
        """Uniform quantization approach

        Notebook: C2S2_DigitalSignalQuantization.ipynb

        Args:
            x: Original signal
            quant_min: Minimum quantization level
            quant_max: Maximum quantization level
            quant_level: Number of quantization levels

        Returns:
            x_quant: Quantized signal
        """
        dither = np.random.uniform(-(quant_max-quant_min)/(2*(quant_level-1)), 
                                    (quant_max-quant_min)/(2*(quant_level-1)),
                                    size=np.array(x).shape) 
        x_normalize = np.array(x) + dither
        x_normalize = (x_normalize-quant_min) * (quant_level-1) / (quant_max-quant_min)
        x_normalize[x_normalize > quant_level - 1] = quant_level - 1
        x_normalize[x_normalize < 0] = 0
        x_normalize_quant = np.around(x_normalize)
        x_quant = (x_normalize_quant) * (quant_max-quant_min) / (quant_level-1) + quant_min
        return tf.convert_to_tensor(x_quant - dither, dtype=tf.float32)

    class VecQuant():
        """
        wrapper class for vector quantizer functions
        """

        def compress(self, X):
            if args.vecdim > 1:
                return quantize_vector(X, quant_level=args.quant_level, dim=args.vecdim), None
            else:
                return quantize_scalar(X, quant_level=args.quant_level), None

        def decompress(self, X, ctx):
            return X 

    class NoneCompressor():
        """Default no-op compression."""

        def compress(self, tensor):
            """Returns the tensor unmodified."""
            return [tensor], None

        def decompress(self, tensors, ctx):
            """Returns the tensor unmodified."""
            #tensor, = tensors
            return tensors
    

    if args.small_part:
        args.save_every = 2**30

    target_repl = (args.target_repl_coef > 0.0 and args.mode == 'train')

    # Build readers, discretizers, normalizers
    train_reader = InHospitalMortalityReader(dataset_dir=os.path.join(args.data, 'train'),
                                            listfile=os.path.join(args.data, 'train_listfile.csv'),
                                            period_length=48.0)

    val_reader = InHospitalMortalityReader(dataset_dir=os.path.join(args.data, 'train'),
                                        listfile=os.path.join(args.data, 'val_listfile.csv'),
                                        period_length=48.0)

    test_reader = InHospitalMortalityReader(dataset_dir=os.path.join(args.data, 'test'),
                                            listfile=os.path.join(args.data, 'test_listfile.csv'),
                                            period_length=48.0)

    discretizer = Discretizer(timestep=float(args.timestep),
                            store_masks=True,
                            impute_strategy='previous',
                            start_time='zero')

    discretizer_header = discretizer.transform(train_reader.read_example(0)["X"])[1].split(',')
    cont_channels = [i for (i, x) in enumerate(discretizer_header) if x.find("->") == -1]

    normalizer = Normalizer(fields=cont_channels)  # choose here which columns to standardize
    normalizer_state = args.normalizer_state
    if normalizer_state is None:
        normalizer_state = 'ihm_ts{}.input_str:{}.start_time:zero.normalizer'.format(args.timestep, args.imputation)
        normalizer_state = os.path.join(os.path.dirname(__file__), normalizer_state)
    normalizer.load_params(normalizer_state)

    args_dict = dict(args._get_kwargs())
    args_dict['header'] = discretizer_header
    args_dict['task'] = 'ihm'
    args_dict['target_repl'] = target_repl
    args_dict['downstream_clients'] = num_clients # total number of vertical partitions present

    models = []
    # Make models for each client
    for i in range(num_clients+1):
        # Build the model
        args.network = "mimic3models/keras_models/lstm"
        if i < num_clients:
            args.network += "_bottom.py"
        else:
            args.network += "_top.py"

        print("==> using model {}".format(args.network))
        model_module = imp.load_source(os.path.basename(args.network), args.network)
        model = model_module.Network(input_dim=int(76/num_clients), **args_dict)

        # Compile the model
        print("==> compiling the model")
        optimizer_config = tf.keras.optimizers.SGD(learning_rate=lr)
        if target_repl:
            loss = ['binary_crossentropy'] * 2
            loss_weights = [1 - args.target_repl_coef, args.target_repl_coef]
        else:
            loss = 'binary_crossentropy'
            loss_weights = None

        model.compile(optimizer=optimizer_config,
                    loss=loss,
                    loss_weights=loss_weights)
        model.summary()
        models.append(model)

    server_model = model_module.Network(input_dim=int(76/num_clients), **args_dict)
    server_model.compile(optimizer=optimizer_config,
                loss=loss,
                loss_weights=loss_weights)

    # Load model weights
    n_trained_chunks = 0
    if args.load_state != "":
        model.load_weights(args.load_state)
        n_trained_chunks = int(re.match(".*epoch([0-9]+).*", args.load_state).group(1))


    """
    Uncomment first block if first time running.
    Use second block after running once for faster startup.
    """
    # Read data from file and save to pickle
    train_raw = utils.load_data(train_reader, discretizer, normalizer, args.small_part)
    pickle.dump(train_raw, open('train_raw1.pkl', 'wb'))
    val_raw = utils.load_data(val_reader, discretizer, normalizer, args.small_part)
    pickle.dump(val_raw, open('val_raw1.pkl', 'wb'))
    test_raw = utils.load_data(test_reader, discretizer, normalizer, args.small_part)
    pickle.dump(test_raw, open('test_raw1.pkl', 'wb'))
    
    # # Read data from pickle
    # train_raw = pickle.load(open('train_raw1.pkl', 'rb'))
    # val_raw = pickle.load(open('val_raw1.pkl', 'rb'))
    # test_raw = pickle.load(open('test_raw1.pkl', 'rb'))    
    
    if target_repl:
        T = train_raw[0][0].shape[0]

        def extend_labels(data):
            data = list(data)
            labels = np.array(data[1])  # (B,)
            data[1] = [labels, None]
            data[1][1] = np.expand_dims(labels, axis=-1).repeat(T, axis=1)  # (B, T)
            data[1][1] = np.expand_dims(data[1][1], axis=-1)  # (B, T, 1)
            return data

        train_raw = extend_labels(train_raw)
        val_raw = extend_labels(val_raw)

    # Prepare training

    print("==> training")

    activation = tf.keras.activations.sigmoid 
    coords_per = int(76/num_clients)

    # Training functions
    def get_grads(x, y, H, model, server_model, i):#, error=False):
        """
        Calculate client embedding, and other client embeddings
        to get the gradient for client model
        """
        loss_value = 0
        Hnew = H.copy()
        with backprop.GradientTape() as tape:
            out = model(x, training=True)
            Hnew[i] = out
            logits = server_model(tf.concat(Hnew,axis=1), training=True)
            loss_value = server_model.compiled_loss(y, logits)
        grads = tape.gradient(loss_value, model.trainable_variables 
                                        + server_model.trainable_variables)
        return grads, loss_value

    def train_step(x, y, model, server_model, H, local, i):
        """
        Traing a client model for 'local' local iterations
        """
        loss_value = 0
        for t in range(local):
            grads, loss_value = get_grads(x, y, H, model, server_model, i)
            grads = model.optimizer._clip_gradients(grads)    # pylint: disable=protected-access
            # only use grads up to the 3rd index. 
            # The last two are for the server model, 
            # which is not necessary as the servermodel is fixed
            model.optimizer.apply_gradients(zip(grads[:3],
                                                model.trainable_variables))
        return loss_value, grads[3][16*i:16*(i+1)]

    def getserver_grads(y, H, server_model):
        """
        Use client embeddings to get current server model gradient
        """
        loss_value = 0
        Hnew = H.copy()
        with backprop.GradientTape() as tape:
            logits = server_model(tf.concat(Hnew,axis=1), training=True)
            loss_value = server_model.compiled_loss(y, logits)
        grads = tape.gradient(loss_value, server_model.trainable_variables)
        return grads, loss_value

    def trainserver_step(y, server_model, H, local):
        """
        Train the server model for 'local' local iterations
        """
        global args
        loss_value = 0
        for t in range(local):
            grads, loss_value = getserver_grads(y, H, server_model)
            grads = server_model.optimizer._clip_gradients(grads)    # pylint: disable=protected-access
            # since we are only getting gradient for the server model trainable variables, 
            # we can just pass in the entire grads list
            server_model.optimizer.apply_gradients(zip(grads,
                                            server_model.trainable_variables))
        return loss_value

    # Get client embeddings
    def forward(x, y, model):
        out = model(x, training=False)
        return out 

    # Get predicted labels to calculating accuracy 
    def predict(x, y, models):
        out = []
        for i in range(len(models)-1):
            x_local = x[:,:,coords_per*i:coords_per*(i+1)]
            out.append(forward(x_local, y, models[i]))
        logits = models[-1](tf.concat(out,axis=1), training=False)
        loss = models[-1].compiled_loss(y, logits)
        return logits , loss

    # Split data into batches 
    train_dataset = tf.data.Dataset.from_tensor_slices((
                                    train_raw[0], 
                                    train_raw[1].reshape(-1,1)))
    train_dataset_static_for_logging = tf.data.Dataset.from_tensor_slices((
                                    train_raw[0], 
                                    train_raw[1].reshape(-1,1)))
    
    train_dataset = train_dataset.batch(args.batch_size)
    train_dataset_static_for_logging = train_dataset_static_for_logging.batch(args.batch_size)

    test_dataset = tf.data.Dataset.from_tensor_slices((
                                    test_raw[0], 
                                    test_raw[1].reshape(-1,1)))
    test_dataset_static_for_logging = tf.data.Dataset.from_tensor_slices((
                                    test_raw[0], 
                                    test_raw[1].reshape(-1,1)))
    
    test_dataset = test_dataset.batch(args.batch_size)
    test_dataset_static_for_logging = test_dataset_static_for_logging.batch(args.batch_size)


    workers = num_clients
    losses = []
    accs_train = []
    accs_test = []

    # Get initial loss and accuracy
    predictions = np.zeros(train_raw[0].shape[0])
    left = 0
    total_loss = 0
    for step, (x_batch_train, y_batch_train) in enumerate(train_dataset_static_for_logging):
        logits, loss_aggregate_model = predict(x_batch_train, y_batch_train, models)
        total_loss += loss_aggregate_model
        predictions[left: left + len(x_batch_train)] = tf.reshape(tf.identity(logits),-1)
        left = left + len(x_batch_train)
    losses.append(total_loss/len(train_dataset_static_for_logging))
    print(f"************Loss = {losses[-1]}***************")
    pickle.dump(losses, open(f'losses_varlr_BS{args.batch_size}_NC{args.num_clients}_LE{args.local_epochs}_Q{args.quant_level}_E{args.correct}_dim{args.vecdim}_comp{args.comp}_seed{args.seed}.pkl', 'wb'))

    # Calculate Training Accuracy 
    ret = metrics.print_metrics_binary(train_raw[1], predictions, verbose=0)
    accs_train.append(list(ret.items()))
    pickle.dump(accs_train, open(f'accs_train_varlr_BS{args.batch_size}_NC{args.num_clients}_LE{args.local_epochs}_Q{args.quant_level}_E{args.correct}_dim{args.vecdim}_comp{args.comp}_seed{args.seed}.pkl', 'wb'))
    print(f"************Train F1-Score = {ret['f1_score']}************")

    # Calculate Test Accuracy 
    predictions = np.zeros(test_raw[0].shape[0])
    left = 0
    for step, (x_batch_test, y_batch_test) in enumerate(test_dataset_static_for_logging):
        logits, _ = predict(x_batch_test, y_batch_test, models)
        predictions[left: left + len(x_batch_test)] = tf.reshape(tf.identity(logits),-1)
        left = left + len(x_batch_test)
    ret = metrics.print_metrics_binary(test_raw[1], predictions, verbose=0)
    accs_test.append(list(ret.items()))
    pickle.dump(accs_test, open(f'accs_test_varlr_BS{args.batch_size}_NC{args.num_clients}_LE{args.local_epochs}_Q{args.quant_level}_E{args.correct}_dim{args.vecdim}_comp{args.comp}_seed{args.seed}.pkl', 'wb'))
    print(f"************Test F1-Score = {ret['f1_score']}************")

    ratio = 0
    if args.quant_level > 0:
        ratio = math.log(args.quant_level,2)/32
    comp = args.comp
    if comp == "topk":
        compressor = NoneCompressor() 
    elif comp == "randomk":
        compressor = NoneCompressor() 
    elif comp == "quantize":
        compressor = VecQuant()
    else:
        compressor = NoneCompressor() 


    grads_Hs = np.empty((num_clients), dtype=object)
    grads_Hs.fill([])
    # Main training loop
    for epoch in range(args.epochs): #tqdm(range(args.epochs)):
        #for i in range(num_clients+1):
        #    models[i].optimizer.lr = 1/(1+epoch)

        #train_dataset = train_dataset.shuffle(buffer_size=train_raw[0].shape[0])
        print("\nStart of epoch %d" % (epoch,))

        Hs = np.empty((math.ceil(train_raw[0].shape[0] / args.batch_size), num_clients), dtype=object)
        Hs.fill([])
        ctx = np.empty((math.ceil(train_raw[0].shape[0] / args.batch_size), num_clients), dtype=object)
        num_batches = 0

        # Iterate over the batches of the dataset.
        for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
            num_batches += 1

            # Exchange client embeddings
            for i in range(num_clients):
                x_local = x_batch_train[:,:,coords_per*i:coords_per*(i+1)]
                H_out = forward(x_local, y_batch_train, models[i])

                # Compress embedding
                if comp == "randomk":
                    # Choose k random indices to send
                    H_tmp = H_out.numpy().reshape(-1)
                    num = int(H_tmp.shape[0]*(1-ratio))
                    indices = torch.randperm(H_tmp.shape[0])[:num]
                    H_tmp[indices] = 0
                    Hs[step,i] = H_tmp.reshape(H_out.shape)
                elif comp == "topk" and not (epoch == 0 and step == 0):
                    # Choose k indices to send based on grads_Hs[i]
                    H_tmp = H_out.numpy()
                    num = math.ceil(H_tmp.shape[1]*ratio)
                    grads = tf.math.abs(tf.reshape(grads_Hs[i],-1))
                    _, indices = tf.math.top_k(grads, 16)
                    H_tmp[:,indices[num:]] = 0
                    Hs[step,i] = H_tmp
                elif comp == "topk":
                    # If first iteration of top-k, do nothing
                    Hs[step,i], ctx[step,i] = copy.deepcopy(H_out), None
                elif comp != "":
                    # If any other compressor (quantize) use the compressor function
                    Hs[step,i], ctx[step,i] = compressor.compress(copy.deepcopy(H_out))
                elif comp == "":
                    # Do nothing
                    Hs[step,i] = copy.deepcopy(H_out)

            # switch to regular top-k compressor for server model compression
            if comp == "topk":
                compressor = TopKCompressor(ratio)

            # Compress the server model
            if comp != "":
                server_model_weights = []
                if comp == "randomk":
                    orig = models[-1].get_weights()[0]
                    H_tmp = tf.transpose(orig).numpy().reshape(-1)
                    num = int(H_tmp.shape[0]*(1-ratio))
                    indices = torch.randperm(H_tmp.shape[0])[:num]
                    H_tmp[indices] = 0
                    decomp = H_tmp.reshape(orig.shape)
                else:
                    compressed, ctx_tmp = compressor.compress(tf.transpose(models[-1].get_weights()[0]))
                    decomp = tf.transpose(compressor.decompress(compressed, ctx_tmp))
                server_model_weights.append(decomp) 
                server_model_weights.append(models[-1].get_weights()[1])
                server_model.set_weights(server_model_weights)
            else:
                server_model = models[-1]

            # Switch back to custom compression for top-k
            if comp == "topk":
                compressor = NoneCompressor() 

            # Train for each client 
            client_losses = [0]*num_clients
            for i in range(num_clients):
                x_local = x_batch_train[:,:,coords_per*i:coords_per*(i+1)]
                H = []
                # Decompress embeddings for client i 
                if comp != "":
                    for j in range(num_clients):
                        H.append(compressor.decompress(copy.deepcopy(Hs[step,j]), ctx[step,j]))
                else:
                    H = copy.deepcopy(Hs[step]).tolist()

                le = local_epochs
                client_losses[i], grads_Hs[i] = train_step(x_local, y_batch_train, models[i], 
                                              server_model, H, le, i)

            # Decompress embeddings for server
            H = []
            if comp != "":
                for j in range(num_clients):
                    H.append(compressor.decompress(copy.deepcopy(Hs[step,j]), ctx[step,j]))
            else:
                H = copy.deepcopy(Hs[step]).tolist()
            # Train server
            loss_final = trainserver_step(y_batch_train, models[-1], H, local_epochs)

        #print("==> predicting")
        # Iterate over the batches of the dataset to calculate loss/accuracy
        # Calculate Training Loss
        predictions = np.zeros(train_raw[0].shape[0])
        left = 0
        total_loss = 0
        for step, (x_batch_train, y_batch_train) in enumerate(train_dataset_static_for_logging):
            logits, loss_aggregate_model = predict(x_batch_train, y_batch_train, models)
            total_loss += loss_aggregate_model
            predictions[left: left + len(x_batch_train)] = tf.reshape(tf.identity(logits),-1)
            left = left + len(x_batch_train)
        losses.append(total_loss/num_batches)
        print(f"************Loss = {losses[-1]}***************")
        pickle.dump(losses, open(f'losses_varlr_BS{args.batch_size}_NC{args.num_clients}_LE{args.local_epochs}_Q{args.quant_level}_E{args.correct}_dim{args.vecdim}_comp{args.comp}_seed{args.seed}.pkl', 'wb'))

        # Calculate Training Accuracy 
        ret = metrics.print_metrics_binary(train_raw[1], predictions, verbose=0)
        accs_train.append(list(ret.items()))
        pickle.dump(accs_train, open(f'accs_train_varlr_BS{args.batch_size}_NC{args.num_clients}_LE{args.local_epochs}_Q{args.quant_level}_E{args.correct}_dim{args.vecdim}_comp{args.comp}_seed{args.seed}.pkl', 'wb'))
        print(f"************Train F1-Score = {ret['f1_score']}************")

        # Calculate Test Accuracy 
        predictions = np.zeros(test_raw[0].shape[0])
        left = 0
        for step, (x_batch_test, y_batch_test) in enumerate(test_dataset_static_for_logging):
            logits, _ = predict(x_batch_test, y_batch_test, models)
            predictions[left: left + len(x_batch_test)] = tf.reshape(tf.identity(logits),-1)
            left = left + len(x_batch_test)
        ret = metrics.print_metrics_binary(test_raw[1], predictions, verbose=0)
        accs_test.append(list(ret.items()))
        pickle.dump(accs_test, open(f'accs_test_varlr_BS{args.batch_size}_NC{args.num_clients}_LE{args.local_epochs}_Q{args.quant_level}_E{args.correct}_dim{args.vecdim}_comp{args.comp}_seed{args.seed}.pkl', 'wb'))
        print(f"************Test F1-Score = {ret['f1_score']}************")
