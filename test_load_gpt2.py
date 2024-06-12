
# %%
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
import os
from torchinfo import summary
# %%
# Initialize wandb
# # Define model and tokenizer
model_name = 'gpt2'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
checkpoint_path = './v2-fine-tuned-gpt2-enwik8-epoch-40'
model = GPT2LMHeadModel.from_pretrained(checkpoint_path)
model.to(device)
tokenizer = GPT2Tokenizer.from_pretrained(checkpoint_path)
# %%