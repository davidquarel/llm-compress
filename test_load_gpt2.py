
# %%
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
import os
# %%
# Initialize wandb
# # Define model and tokenizer
model_name = 'gpt2'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
checkpoint_path = './v2-fine-tuned-gpt2-enwik8-epoch-39'
model = GPT2LMHeadModel.from_pretrained(checkpoint_path)


# %%
def round_clip(x, min_val, max_val):
    return torch.clamp(torch.round(x), min_val, max_val)

def absmean_quantization(weights, epsilon=1e-6):
    gamma = torch.mean(torch.abs(weights))
    quantized_weights = round_clip(weights / (gamma + epsilon), -1, 1)
    return quantized_weights

for name, param in model.named_parameters():
    quantized_weights = absmean_quantization(param.data)  # Quantize and convert to fp16
    param.data = quantized_weights
# %%
