import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
import numpy as np

def run_mwrl_evaluation(model, criterion, train_loader, val_loader, device, writer, epoch):
    """
    Evaluate MWRL model performance across all nesting dimensions.
    """
    model.eval()
    criterion.eval()
    
    # Validation accuracies for each dimension
    val_accuracies = {}
    
    with torch.no_grad():
        # Evaluate on validation set
        total_samples = 0
        correct_predictions = {f"dim_{dim}": 0 for dim in criterion.nesting_list}
        
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            embeddings = model(inputs)
            nesting_logits = criterion.classifiers(embeddings)
            
            # Calculate accuracy for each dimension
            batch_size = targets.size(0)
            total_samples += batch_size
            
            for i, (dim, logits) in enumerate(zip(criterion.nesting_list, nesting_logits)):
                _, predicted = torch.max(logits, 1)
                correct_predictions[f"dim_{dim}"] += (predicted == targets).sum().item()
        
        # Calculate accuracies
        for dim in criterion.nesting_list:
            accuracy = correct_predictions[f"dim_{dim}"] / total_samples
            val_accuracies[f"dim_{dim}"] = accuracy
            
            # Log to tensorboard
            writer.add_scalar(f'Accuracy/val_dim_{dim}', accuracy, epoch)
            print(f'Validation Accuracy (dim {dim}): {accuracy:.4f}')
    
    # Calculate average accuracy
    avg_accuracy = np.mean(list(val_accuracies.values()))
    writer.add_scalar('Accuracy/val_average', avg_accuracy, epoch)
    print(f'Average Validation Accuracy: {avg_accuracy:.4f}')
    
    return avg_accuracy