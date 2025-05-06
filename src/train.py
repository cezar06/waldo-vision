import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import os
import logging
import argparse
from tqdm import tqdm # For progress bars
import numpy as np

# Assuming data_loader.py and model.py are in the same directory (src)
from data_loader import get_data_loaders, DATA_DIR, WALDO_PATH_REL, NOTWALDO_PATH_REL, IMG_SIZE
from model import SimpleCNN, get_finetuned_resnet18

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {DEVICE}")

# --- Helper Functions ---
def calculate_metrics(preds, labels):
    """Calculates accuracy, precision, recall, and F1 score."""
    # Apply sigmoid and threshold to get binary predictions
    preds_binary = (torch.sigmoid(preds) > 0.5).cpu().numpy()
    labels_binary = labels.cpu().numpy()

    accuracy = accuracy_score(labels_binary, preds_binary)
    # Calculate metrics, handle zero division for classes not present
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels_binary, preds_binary, average='binary', zero_division=0
    )
    # Also get metrics per class if needed (useful for imbalance)
    # precision_p, recall_p, f1_p, _ = precision_recall_fscore_support(
    #     labels_binary, preds_binary, average=None, zero_division=0
    # )
    # conf_matrix = confusion_matrix(labels_binary, preds_binary)

    return accuracy, precision, recall, f1

def get_pos_weight(train_loader):
    """Calculate pos_weight for BCEWithLogitsLoss to handle class imbalance."""
    num_negatives = 0
    num_positives = 0
    logging.info("Calculating pos_weight for loss function...")
    # Temporarily disable shuffling to iterate consistently if needed, though counts should be stable
    # Note: This iterates through the entire training set once before training starts.
    # Consider calculating this based on file counts if dataset is huge.
    for _, labels in tqdm(train_loader, desc="Counting classes"):
        num_positives += torch.sum(labels == 1).item()
        num_negatives += torch.sum(labels == 0).item()

    if num_positives == 0:
        logging.warning("No positive samples found in training data! Using pos_weight=1.")
        return torch.tensor([1.0], device=DEVICE)

    pos_weight = num_negatives / num_positives
    logging.info(f"Calculated pos_weight: {pos_weight:.2f} ({num_negatives} negatives / {num_positives} positives)")
    return torch.tensor([pos_weight], device=DEVICE)


# --- Training and Validation Functions ---
def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """Trains the model for one epoch."""
    model.train()  # Set model to training mode
    running_loss = 0.0
    all_preds = []
    all_labels = []

    # Wrap dataloader with tqdm for a progress bar
    progress_bar = tqdm(dataloader, desc="Training", leave=False)
    for inputs, labels in progress_bar:
        # Skip batch if collate_fn returned empty tensors
        if inputs.numel() == 0:
            logging.warning("Skipping empty batch in training.")
            continue
            
        inputs, labels = inputs.to(device), labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        # Statistics
        running_loss += loss.item() * inputs.size(0)
        all_preds.append(outputs.detach()) # Detach from graph
        all_labels.append(labels.detach())

        # Update progress bar description
        progress_bar.set_postfix(loss=loss.item())

    epoch_loss = running_loss / len(dataloader.dataset)
    
    # Concatenate all predictions and labels for metric calculation
    all_preds_tensor = torch.cat(all_preds)
    all_labels_tensor = torch.cat(all_labels)
    epoch_acc, epoch_prec, epoch_rec, epoch_f1 = calculate_metrics(all_preds_tensor, all_labels_tensor)

    return epoch_loss, epoch_acc, epoch_prec, epoch_rec, epoch_f1


def validate(model, dataloader, criterion, device):
    """Validates the model on the validation set."""
    model.eval()  # Set model to evaluation mode
    running_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():  # Disable gradient calculations
        progress_bar = tqdm(dataloader, desc="Validation", leave=False)
        for inputs, labels in progress_bar:
            # Skip batch if collate_fn returned empty tensors
            if inputs.numel() == 0:
                logging.warning("Skipping empty batch in validation.")
                continue

            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            all_preds.append(outputs)
            all_labels.append(labels)

    epoch_loss = running_loss / len(dataloader.dataset)
    
    # Concatenate all predictions and labels for metric calculation
    all_preds_tensor = torch.cat(all_preds)
    all_labels_tensor = torch.cat(all_labels)
    epoch_acc, epoch_prec, epoch_rec, epoch_f1 = calculate_metrics(all_preds_tensor, all_labels_tensor)

    return epoch_loss, epoch_acc, epoch_prec, epoch_rec, epoch_f1

# --- Main Training Loop ---
def main(args):
    logging.info("Starting training process...")

    # Create model save directory if it doesn't exist
    model_save_dir = os.path.join(args.save_dir, args.exp_name)
    os.makedirs(model_save_dir, exist_ok=True)
    logging.info(f"Models will be saved in: {model_save_dir}")

    # Load data
    logging.info("Loading data...")
    train_loader, val_loader = get_data_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    if not train_loader or not val_loader:
        logging.error("Failed to get data loaders. Exiting.")
        return

    # Initialize model - USE THE NEW RESNET MODEL
    logging.info("Initializing fine-tuned ResNet18 model...")
    model = get_finetuned_resnet18(use_pretrained=True).to(DEVICE)
    # model = SimpleCNN().to(DEVICE) # Keep this commented out for reference

    # Define loss function (with pos_weight for imbalance) and optimizer
    pos_weight_tensor = get_pos_weight(train_loader)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Learning rate scheduler (optional, consider if needed after initial runs)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    # Training loop
    best_val_loss = float('inf')
    epochs_no_improve = 0
    patience = 5 # Number of epochs to wait for improvement before stopping

    logging.info(f"Starting training for {args.epochs} epochs.")
    for epoch in range(args.epochs):
        logging.info(f"--- Epoch {epoch+1}/{args.epochs} ---")

        train_loss, train_acc, train_prec, train_rec, train_f1 = train_one_epoch(
            model, train_loader, criterion, optimizer, DEVICE
        )
        logging.info(f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, Prec: {train_prec:.4f}, Rec: {train_rec:.4f}, F1: {train_f1:.4f}")

        val_loss, val_acc, val_prec, val_rec, val_f1 = validate(
            model, val_loader, criterion, DEVICE
        )
        logging.info(f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, Prec: {val_prec:.4f}, Rec: {val_rec:.4f}, F1: {val_f1:.4f}")

        # Optional: Adjust learning rate
        # scheduler.step()

        # Save the best model based on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            best_model_path = os.path.join(model_save_dir, 'best_model.pth')
            torch.save(model.state_dict(), best_model_path)
            logging.info(f"Validation loss improved. Saved best model to {best_model_path}")
        else:
            epochs_no_improve += 1
            logging.info(f"Validation loss did not improve. Best loss: {best_val_loss:.4f} ({epochs_no_improve}/{patience})")

        # Early stopping
        if epochs_no_improve >= patience:
            logging.info(f"Early stopping triggered after {epoch + 1} epochs.")
            break
            
    # Save the final model state
    final_model_path = os.path.join(model_save_dir, 'final_model.pth')
    torch.save(model.state_dict(), final_model_path)
    logging.info(f"Training finished. Saved final model to {final_model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a Waldo classification model.')
    parser.add_argument('--data-dir', type=str, default=DATA_DIR, help='Path relative to the script location for the data directory')
    parser.add_argument('--epochs', type=int, default=15, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size for training and validation')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate (often lower for fine-tuning)')
    parser.add_argument('--num-workers', type=int, default=0, help='Number of workers for DataLoader')
    parser.add_argument('--save-dir', type=str, default='../models', help='Directory relative to script to save trained models')
    parser.add_argument('--exp-name', type=str, default='waldo_classifier_resnet18', help='Subdirectory name for this experiment\'s models')

    args = parser.parse_args()
    main(args) 