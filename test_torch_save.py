# %%
import torch
import torch.nn.functional as F
# %%
input_file = './tokens_enwik8_raw.pt'
chunks = torch.load(input_file)
print(f"Loaded tokenized chunks from {input_file}")
# %%
def sliding_windows(tensor, window_size, step_size):
    # Calculate the shape of the resulting tensor
    num_windows = (tensor.size(0) - window_size) // step_size + 1
    # Calculate the strides
    stride = tensor.stride(0)
    new_stride = (step_size * stride, stride)
    # Use as_strided to create the sliding windows
    windows = torch.as_strided(tensor, size=(num_windows, window_size), stride=new_stride)
    return windows
# %%
