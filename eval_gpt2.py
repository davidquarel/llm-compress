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
avg_loss, accuracy = utils.evaluate_model(model, dataloader)
print(f"Average Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
# %%
