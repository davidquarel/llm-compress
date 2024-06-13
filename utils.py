import torch
from tqdm import tqdm
import os

def sliding_windows(tensor, window_size, step_size):
    # Calculate the shape of the resulting tensor
    num_windows = (tensor.size(0) - window_size) // step_size + 1
    # Calculate the strides
    stride = tensor.stride(0)
    new_stride = (step_size * stride, stride)
    # Use as_strided to create the sliding windows
    windows = torch.as_strided(tensor, size=(num_windows, window_size), stride=new_stride)
    return windows

def preprocess_tokens(chunks, model, tokenizer):
    """
    Gives a dataset where the context window is half-full
    Model has to predict what remains
    """
    window_size = model.config.n_ctx - 1
    stride = (model.config.n_ctx // 2) - 1
    batched = sliding_windows(chunks, window_size, stride)
    bos_tokens = torch.tensor([tokenizer.bos_token_id] * batched.size(0)).unsqueeze(1)
    input_ids = torch.cat([bos_tokens, batched], dim=1)
    return input_ids

def find_latest_checkpoint(base_path):
    checkpoints = [path for path in os.listdir(base_path) if "epoch-" in path]
    if not checkpoints:
        return None
    latest_checkpoint = max(checkpoints, key=lambda x: int(x.split('-')[-1]))
    return os.path.join(base_path, latest_checkpoint)

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