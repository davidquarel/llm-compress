# %%
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from torch.utils.data import DataLoader
from tqdm import tqdm
import utils

torch.set_grad_enabled(False)
# Define model and tokenizer
model_name = 'gpt2'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
checkpoint = "./v2-fine-tuned-gpt2-enwik8-epoch-39"
model = GPT2LMHeadModel.from_pretrained(checkpoint)
print("dtype: ", model.parameters().__next__().dtype)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
import numpy as np
# %%
# Load the data
input_file = './tokens_enwik8_raw.pt'
chunks = torch.load(input_file)

input_ids = utils.preprocess_tokens(chunks, model, tokenizer).to(device)

# DataLoader setup
batch_size = 16
dataloader = DataLoader(input_ids, batch_size=batch_size, shuffle=False)

@torch.no_grad()
def evaluate_model(model, dataloader):
    model.eval()
    device = next(model.parameters()).device
    total_loss = 0
    total_correct = 0
    total_correct_5 = 0
    total_processed = 0
    loss_fct = torch.nn.CrossEntropyLoss()
    progress_bar = tqdm(dataloader, desc="Evaluating")

    for input_ids in progress_bar:
        input_ids = input_ids.to(device)
        context_size = 512
        target_tokens = input_ids[:, context_size:].flatten()
    
        logits = model(input_ids).logits
        logits = logits[:, context_size-1:-1].reshape(input_ids.shape[0] * context_size, -1)
        
        loss = loss_fct(logits, target_tokens)
        total_loss += loss.item() * target_tokens.numel()
        
        preds = logits.argmax(dim=-1)
        correct = (preds == target_tokens).sum()
        top5_preds = torch.topk(logits, k=5, dim=-1)[1]
        top5_correct = (top5_preds == target_tokens.unsqueeze(-1)).any(dim=-1).sum()
        total_correct += correct.item()
        total_correct_5 += top5_correct.item()
        total_processed += target_tokens.numel()

        # Calculate running average loss and accuracy
        avg_loss = total_loss / total_processed
        accuracy = total_correct / total_processed
        acc_5 = total_correct_5 / total_processed
        
        # Update progress bar description with current average loss and accuracy
        progress_bar.set_description(f"loss: {avg_loss:.4f}, acc: {accuracy:.4f}, acc5: {acc_5:.4f}")
        #progress_bar.set_postfix(loss=avg_loss, correct=total_correct, total=total_processed, accuracy=accuracy)

    avg_loss = total_loss / total_processed
    accuracy = total_correct / total_processed
    return avg_loss, accuracy

# Evaluate the model

# def round_clip(x, min_val, max_val):
#     return torch.clamp(torch.round(x), min_val, max_val)

# def absmean_quantization(weights, epsilon=1e-6):
#     gamma = torch.mean(torch.abs(weights))
#     quantized_weights = round_clip(weights / (gamma + epsilon), -1, 1)
#     return quantized_weights

# for name, param in model.named_parameters():
#     if 'weight' in name:
#         quantized_weights = absmean_quantization(param.data)  # Quantize and convert to fp16
#         param.data = quantized_weights
# %%
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, GemmaTokenizer, GPTQConfig

model = GPT2LMHeadModel.from_pretrained(checkpoint)

gptq_config = GPTQConfig(bits=2, dataset="wikitext2", tokenizer=tokenizer)
quantized_model = AutoModelForCausalLM.from_pretrained(checkpoint, 
                                                       quantization_config=gptq_config)
# %%
torch.save(model.state_dict(), "model_quantized.pth")
# %%    
#     # Create 16 quantization levels
#     quant_levels = np.linspace(min_val, max_val, 16)
    
#     # Discretize weights to the nearest quantization level
#     quantized_weights = np.digitize(weights.cpu().numpy(), quant_levels, right=True)
#     quantized_weights = quant_levels[quantized_weights - 1]  # Adjust indices
    
#     return torch.tensor(quantized_weights, dtype=weights.dtype, device=weights.device)

# def discretize_model_weights(model):
#     for name, param in model.named_parameters():
#         if param.requires_grad:
#             param.data = discretize_weights_to_4bit(param.data)
# # %%
model.to(device).half()
avg_loss, accuracy = evaluate_model(model, dataloader)
print(f"Average Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
# %%
