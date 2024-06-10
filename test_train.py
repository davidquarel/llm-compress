# %%
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from torch.optim import AdamW
from torch.utils.data import DataLoader
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

# Load the tokenized chunks from disk
input_file = '/root/tokenized_enwik8.pt'
chunks = torch.stack(torch.load(input_file), dim=0)
print(f"Loaded tokenized chunks from {input_file}")

# Create DataLoader
batch_size = 4  # Adjust this value as needed
train_dataloader = DataLoader(chunks, batch_size=batch_size, shuffle=True)

# Define optimizer
learning_rate = 1e-3  # High learning rate for overfitting
optimizer = AdamW(model.parameters(), lr=learning_rate)

# Function to train one epoch
def train_one_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    progress_bar = tqdm(dataloader, desc="Training")
    for batch in progress_bar:
        input_ids = batch.to(device)
        labels = batch.clone().to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item())

        # Log batch loss to wandb
        wandb.log({"batch_loss": loss.item()})

    avg_loss = total_loss / len(dataloader)
    return avg_loss

# Training loop
num_epochs = 10  # Increase number of epochs for overfitting
for epoch in range(num_epochs):
    avg_epoch_loss = train_one_epoch(model, train_dataloader, optimizer, device)
    print(f"Average loss for epoch {epoch+1}: {avg_epoch_loss}")

    # Log average epoch loss to wandb
    wandb.log({"avg_epoch_loss": avg_epoch_loss})

    # Save checkpoint after each epoch
    checkpoint_path = f"fine-tuned-gpt2-enwik8-epoch-{epoch+1}"
    model.save_pretrained(checkpoint_path)
    tokenizer.save_pretrained(checkpoint_path)
    print(f"Checkpoint saved to '{checkpoint_path}'")

# Save the final model
model.save_pretrained("fine-tuned-gpt2-enwik8-final")
tokenizer.save_pretrained("fine-tuned-gpt2-enwik8-final")

print("Model fine-tuning complete and saved to 'fine-tuned-gpt2-enwik8-final'.")

# %%
