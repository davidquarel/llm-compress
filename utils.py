import torch

def sliding_windows(tensor, window_size, step_size):
    # Calculate the shape of the resulting tensor
    num_windows = (tensor.size(0) - window_size) // step_size + 1
    # Calculate the strides
    stride = tensor.stride(0)
    new_stride = (step_size * stride, stride)
    # Use as_strided to create the sliding windows
    windows = torch.as_strided(tensor, size=(num_windows, window_size), stride=new_stride)
    return windows

def preprocess_tokens(chunks, model, tokenizer):
    """
    Gives a dataset where the context window is half-full
    Model has to predict what remains
    """
    window_size = model.config.n_ctx - 1
    stride = (model.config.n_ctx // 2) - 1
    batched = sliding_windows(chunks, window_size, stride)
    bos_tokens = torch.tensor([tokenizer.bos_token_id] * batched.size(0)).unsqueeze(1)
    input_ids = torch.cat([bos_tokens, batched], dim=1)
    return input_ids

def find_latest_checkpoint(base_path):
    checkpoints = [path for path in os.listdir(base_path) if "epoch-" in path]
    if not checkpoints:
        return None
    latest_checkpoint = max(checkpoints, key=lambda x: int(x.split('-')[-1]))
    return os.path.join(base_path, latest_checkpoint)

