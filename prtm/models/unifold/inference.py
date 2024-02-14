import math

import torch


def get_device_mem(device):
    if device != "cpu" and torch.cuda.is_available():
        cur_device = torch.cuda.current_device()
        prop = torch.cuda.get_device_properties("cuda:{}".format(cur_device))
        total_memory_in_GB = prop.total_memory / 1024 / 1024 / 1024
        return total_memory_in_GB
    else:
        return 40


def automatic_chunk_size(seq_len, device, is_bf16):
    total_mem_in_GB = get_device_mem(device)
    factor = math.sqrt(total_mem_in_GB / 40.0 * (0.55 * is_bf16 + 0.45)) * 0.95
    if seq_len < int(1024 * factor):
        chunk_size = 256
        block_size = None
    elif seq_len < int(2048 * factor):
        chunk_size = 128
        block_size = None
    elif seq_len < int(3072 * factor):
        chunk_size = 64
        block_size = None
    elif seq_len < int(4096 * factor):
        chunk_size = 32
        block_size = 512
    else:
        chunk_size = 4
        block_size = 256
    return chunk_size, block_size
