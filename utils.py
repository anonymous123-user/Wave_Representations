import torch
import numpy as np
import math
import imageio
import random
import pdb


def set_random_seed(seed):
    """function sets the seed value
    Args:
        seed (int): seed value
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_model(device, block_type, model_type, oscillator_type, num_classes,
               N, M, dt1, min_iters, max_iters, c_in, c_out, 
               rnn_kernel, num_blocks, num_slots, num_iters, img_size, kernel_init, cell_type, num_layers):
    if model_type == 'cornn_model':
        from cornn_model import Model
        net = Model(N, c_in, c_out, num_classes, min_iters, max_iters, img_size, num_slots, dt1, max_iters)
    elif model_type == 'baseline2':
        from baseline2 import Model
        net = Model(N, c_in, c_out, num_classes, min_iters, max_iters, img_size, num_slots, dt1, max_iters, cell_type)
    elif model_type == 'baseline2_fft':
        from baseline2_fft import Model
        net = Model(N, c_in, c_out, num_classes, min_iters, max_iters, img_size, num_slots, dt1, max_iters, cell_type)
    elif model_type == 'baseline3':
        from baseline3 import Model
        net = Model(N, c_in, c_out, num_classes, min_iters, max_iters, img_size, num_slots, dt1, max_iters, cell_type)
    elif model_type == 'baseline3_fft':
        from baseline3_fft import Model
        net = Model(N, c_in, c_out, num_classes, min_iters, max_iters, img_size, num_slots, dt1, max_iters, cell_type)
    elif model_type == 'baseline1_flexible':
        from baseline1_flexible import Model
        net = Model(N, c_in, c_out, num_classes, min_iters, max_iters, img_size, num_slots, dt1, max_iters, num_layers)
    elif model_type == 'unet':
        from unet import UNet as Model
        net = Model(c_in, num_classes)
    net = net.to(device)
    return net

#def build_encoder(c_in, c_out, num_layers):



def load_block(block_type, oscillator_type, N, M, dt, 
               min_iters, max_iters, c_in, c_out, rnn_kernel, kernel_init, num_slots):
    if block_type == 'block1':
        from block1 import Block

    oscillator = load_oscillator(block_type, oscillator_type, N, M, dt, 
                    min_iters, max_iters, c_in, c_out, rnn_kernel, kernel_init, num_slots)

    return Block(oscillator, oscillator_type, N, M, dt, min_iters, max_iters, c_in,
                 c_out, rnn_kernel, kernel_init, num_slots)


def load_oscillator(block_type, oscillator_type, N, M, dt, 
                    min_iters, max_iters, c_in, c_out, rnn_kernel, kernel_init, num_slots):
    if oscillator_type == 'cornn':
        from cornn_oscillator import Oscillator   
        return Oscillator(channels=c_in, N=N, dt=dt, iters=max_iters, kernel_size=rnn_kernel, hidden_channels=c_out, kernel_init=kernel_init)
    else:
        print(f"ERROR: {oscillator_type} is an invalid oscillator type.")


def save_weights(weights, cp_path):
    torch.save(weights, cp_path)


def save_model(net, cp_path):
    torch.save(net.state_dict(), cp_path)


def load_model(net, cp_path):
    net.load_state_dict(torch.load(cp_path), strict=False)
    net.eval()
    return net