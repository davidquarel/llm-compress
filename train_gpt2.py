# %%
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
import os
#===
import utils


# Initialize wandb
wandb.init(project="gpt2-enwik8-finetuning")

# Define model and tokenizer
model_name = 'gpt2'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GPT2LMHeadModel.from_pretrained(model_name)
model.to(device)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
# %%

input_file = './tokens_enwik8_raw.pt'
chunks = torch.load(input_file)
print(f"Loaded tokenized chunks from {input_file}")

input_ids = utils.preprocess_tokens(chunks, model, tokenizer).to(device)

# %%
# Create DataLoader
batch_size = 4  # Adjust this value as needed
train_dataloader = DataLoader(input_ids, batch_size=batch_size, shuffle=True)

# Define optimizer
learning_rate = 1e-3  # High learning rate for overfitting
optimizer = AdamW(model.parameters(), lr=learning_rate)
# %%
# Function to train one epoch
def train_one_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    progress_bar = tqdm(dataloader, desc="Training")
    for input_ids in progress_bar:
        
        # Only consider the loss for the last 512 tokens
        context_size = 512
        target_tokens = input_ids[:, context_size:]
        optimizer.zero_grad()
        logits = model(input_ids).logits
        logits = logits[:, context_size-1:-1].reshape(input_ids.shape[0] * context_size, -1)
        loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(logits, target_tokens.flatten())
        
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item())

        # Log batch loss to wandb
        wandb.log({"batch_loss": loss.item()})

    avg_loss = total_loss / len(dataloader)
    return avg_loss
# %%
# Function to find the latest checkpoint in the directory


# Base directory where the checkpoints are saved
base_checkpoint_dir = "./gptv2"

# Check for the latest checkpoint
latest_checkpoint = utils.find_latest_checkpoint(base_checkpoint_dir)

# Load the model and tokenizer from the latest checkpoint if exists
if latest_checkpoint:
    print(f"Resuming training from {latest_checkpoint}")
    model = model.from_pretrained(latest_checkpoint)
    #tokenizer = tokenizer.from_pretrained(latest_checkpoint)
    # Assuming the optimizer state is also saved and needs to be loaded
    #optimizer.load_state_dict(torch.load(os.path.join(latest_checkpoint)))
    start_epoch = int(latest_checkpoint.split('-')[-1])
else:
    print("No checkpoint found, starting from scratch")
    start_epoch = 0
    
model.to(device)
# %%


# Training loop
num_epochs = 1000  # Increase number of epochs for overfitting
for epoch in range(start_epoch, num_epochs):
    avg_epoch_loss = train_one_epoch(model, train_dataloader, optimizer, device)
    print(f"Average loss for epoch {epoch+1}: {avg_epoch_loss}")

    # Log average epoch loss to wandb
    wandb.log({"avg_epoch_loss": avg_epoch_loss})

    # Save checkpoint after each epoch
    checkpoint_path = f"v2-fine-tuned-gpt2-enwik8-epoch-{epoch+1}"
    model.save_pretrained(checkpoint_path)
    tokenizer.save_pretrained(checkpoint_path)
    print(f"Checkpoint saved to '{checkpoint_path}'")

# Save the final model
model.save_pretrained("fine-tuned-gpt2-enwik8-final")
tokenizer.save_pretrained("fine-tuned-gpt2-enwik8-final")

print("Model fine-tuning complete and saved to 'fine-tuned-gpt2-enwik8-final'.")

# %%
