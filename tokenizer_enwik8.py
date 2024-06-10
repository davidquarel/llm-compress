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

# Chunk the tokenized text into segments of size 1023
chunk_size = 1023
chunks = [tokenized_text[i:i + chunk_size] for i in tqdm(range(0, len(tokenized_text), chunk_size))]

# Prepend each chunk with the <bos> token
bos_token_id = tokenizer.bos_token_id
chunks = [torch.cat((torch.tensor([bos_token_id]), chunk), dim=0) for chunk in chunks if len(chunk) == chunk_size]

# Save the chunks to a file
output_file = 'tokenized_enwik8.pt'
torch.save(chunks, output_file)
print(f"Tokenized chunks saved to {output_file}")
