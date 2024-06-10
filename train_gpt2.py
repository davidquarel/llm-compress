# %%
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AdamW, get_scheduler
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import wandb

# Initialize wandb
wandb.init(project="gpt2-enwik8-finetuning")

# Define model and tokenizer
model_name = 'gpt2'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GPT2LMHeadModel.from_pretrained(model_name)
model.to(device)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Add the <bos> token if not present
if tokenizer.bos_token is None:
    tokenizer.add_special_tokens({'bos_token': '<|endoftext|>'})
    model.resize_token_embeddings(len(tokenizer))

# Load enwik8 dataset
dataset = load_dataset('enwik8', split='train')

# Concatenate the entire text
full_text = ''.join(dataset['text'])

# Tokenize the entire text
tokenized_text = tokenizer(full_text, return_tensors='pt')['input_ids'][0]

# Chunk the tokenized text into segments of size 1023
chunk_size = 1023
chunks = [tokenized_text[i:i + chunk_size] for i in range(0, len(tokenized_text), chunk_size)]

# Prepend each chunk with the <bos> token
bos_token_id = tokenizer.bos_token_id
chunks = [torch.cat((torch.tensor([bos_token_id]), chunk), dim=0) for chunk in chunks if len(chunk) == chunk_size]

# Custom Dataset to handle the encoded chunks
class Enwik8Dataset(Dataset):
    def __init__(self, chunks):
        self.input_ids = torch.stack(chunks)
        
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': torch.ones_like(self.input_ids[idx])
        }

# Create DataLoader
train_dataset = Enwik8Dataset(chunks)
train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)
# %%
# Define optimizer and scheduler
optimizer = AdamW(model.parameters(), lr=5e-5)
num_training_steps = len(train_dataloader) * 3  # 3 epochs
lr_scheduler = get_scheduler(
    name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
)

# Set up wandb config
wandb.config.update({
    "model_name": model_name,
    "batch_size": 2,
    "learning_rate": 5e-5,
    "epochs": 3,
})

# Set up the training loop
model.train()
for epoch in range(3):  # Number of epochs
    progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}")
    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = input_ids.clone().to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        progress_bar.set_postfix(loss=loss.item())

        # Log metrics to wandb
        wandb.log({"loss": loss.item()})

# Save the fine-tuned model
model.save_pretrained("fine-tuned-gpt2-enwik8")
tokenizer.save_pretrained("fine-tuned-gpt2-enwik8")

print("Model fine-tuning complete and saved to 'fine-tuned-gpt2-enwik8'.")
