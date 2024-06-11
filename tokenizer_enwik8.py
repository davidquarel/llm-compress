import torch
from transformers import GPT2Tokenizer
from datasets import load_dataset
from tqdm import tqdm

# Define tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Add the <bos> token if not present
if tokenizer.bos_token is None:
    tokenizer.add_special_tokens({'bos_token': '<|endoftext|>'})

# Load enwik8 dataset
dataset = load_dataset('enwik8', split='train')

# Concatenate the entire text
full_text = ''.join(dataset['text'])

# Tokenize the entire text with a progress bar
print("Tokenizing the text...")
tokenized_text = tokenizer(full_text, return_tensors='pt')['input_ids'][0]


# Save the chunks to a file
output_file = 'tokens_enwik8_raw.pt'
torch.save(tokenized_text, output_file)
print(f"Tokenized chunks saved to {output_file}")
