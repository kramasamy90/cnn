import torch
import torch.nn.functional as F

def get_top_k_accuracy(y_pred, y_true, k=5):
    top_k_preds = torch.topk(y_pred, k, dim=1).indices
    correct = torch.any(top_k_preds == y_true.unsqueeze(1), dim=1)
    top_k_accuracy = correct.float().mean().item() * 100
    return top_k_accuracy

def evaluate_model(model, dataloader, device='cuda'):
    model.eval()
    y_pred_list = []
    y_true_list = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            logits = model(inputs)
            y_pred_list.append(logits)
            y_true_list.append(labels)

    y_pred = torch.cat(y_pred_list, dim=0)
    y_true = torch.cat(y_true_list, dim=0)

    return y_pred, y_true
