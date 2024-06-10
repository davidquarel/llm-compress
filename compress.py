# %%
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
model_name = 'google/gemma-2b'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = AutoModelForCausalLM.from_pretrained(model_name)
model.to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)
from tqdm import tqdm
from datasets import load_dataset
import numpy as np
# %%
torch.set_grad_enabled(False)
# %%
# Load the enwik8 dataset
dataset = load_dataset('enwik8')
# %%
# Take a subset of the dataset for demonstration purposes
# You can adjust the number of examples based on your requirements
subset = dataset['train'][:10000]  # Example: first 1000 characters

# Convert the subset to a string
text = "".join(subset['text'])

# %%
# Tokenize the text

# %%

def test(ctx_len):

    inputs = tokenizer(text, return_tensors='pt')['input_ids'][0]
    inputs = inputs[:((inputs.shape[0] // ctx_len-1) * (ctx_len-1))]
    inputs = torch.reshape(inputs, (-1, ctx_len-1))
    x = torch.zeros((inputs.shape[0], ctx_len), dtype=torch.int64)
    x[:, 0] = 2
    x[:, 1:] = inputs
    batch_size = 1
    batches = torch.split(x, batch_size, dim=0)


    rank_counts = []
    total_error = 0
    model.eval()
    avg_error = 0
    counts = 0
    runner = tqdm(batches)
    for batch in runner:
        batch = batch.to(device)
        logits = model(batch.to(device)).logits.detach()
        error = torch.nn.CrossEntropyLoss()(logits[:, :-1].reshape(-1, logits.shape[-1]), batch[:, 1:].reshape(-1))
        total_error += error.item()
        counts += 1
        avg_error = total_error / counts
        
        # Get predictions and targets
        
        # Get predictions and targets
        targets = batch[:, 1:]

        # Get the top 20 predictions for each token
        top_k = 20
        topk_values, topk_indices = torch.topk(logits[:, :-1], top_k, dim=-1, largest=True, sorted=True)
        
        # Find the rank of the correct answer for each token within the top 20 predictions
        target_expanded = targets.unsqueeze(-1).expand_as(topk_indices)
        target_ranks = (topk_indices == target_expanded).nonzero(as_tuple=True)[-1]
        
        rank_counts.extend(target_ranks.tolist())

        # Filter for top 5 ranks for histogram
        hist, _ = np.histogram(rank_counts, bins=np.arange(21))

        description = f"Average Error: {avg_error:.4f} | Histogram: {hist}"
        runner.set_description(description)
    return avg_error, rank_counts
    
    
    
    
# %%
results = {}
for ctx_len in [32, 64, 128, 256, 512, 1024, 2048, 4196, 8192]:
    results[ctx_len] = test(ctx_len)
# %%
