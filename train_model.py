import numpy as np
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import gc

from particle_detection import UNet3D, PatchDataset, CombinedLoss

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=50, patience=10):
    """
    Train the 3D detection model

    Parameters:
    model (nn.Module): Model to train
    train_loader (DataLoader): Training data loader
    val_loader (DataLoader): Validation data loader
    criterion (nn.Module): Loss function
    optimizer (optim.Optimizer): Optimizer
    scheduler (optim.lr_scheduler): Learning rate scheduler
    num_epochs (int): Maximum number of epochs
    patience (int): Early stopping patience

    Returns:
    model: Trained model
    history: Training history
    """
    # Initialize variables
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': []}

    # Get device
    device = next(model.parameters()).device

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        # Training step
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for patches, labels in progress_bar:
            # Move data to device
            patches = patches.to(device)
            labels = labels.to(device)

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(patches)

            # Calculate loss
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Update training loss
            train_loss += loss.item() * patches.size(0)

            # Update progress bar
            progress_bar.set_postfix(loss=loss.item())

        # Calculate average training loss
        train_loss /= len(train_loader.dataset)

        # Validation step
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for patches, labels in val_loader:
                # Move data to device
                patches = patches.to(device)
                labels = labels.to(device)

                # Forward pass
                outputs = model(patches)

                # Calculate loss
                loss = criterion(outputs, labels)

                # Update validation loss
                val_loss += loss.item() * patches.size(0)

        # Calculate average validation loss
        val_loss /= len(val_loader.dataset)

        # Update learning rate
        scheduler.step(val_loss)

        # Print progress
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        # Check for improvement
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

        # Free up memory
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model, history

def plot_training_history(history, save_path=None):
    """
    Plot training history

    Parameters:
    history (dict): Training history
    save_path (str, optional): Path to save the plot
    """
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training History')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)

    plt.show()

def main(data_dir, model_dir):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load training and validation data
    train_patches = np.load(os.path.join(data_dir, 'training', 'train_patches.npy'))
    train_labels = np.load(os.path.join(data_dir, 'training', 'train_labels.npy'))
    val_patches = np.load(os.path.join(data_dir, 'training', 'val_patches.npy'))
    val_labels = np.load(os.path.join(data_dir, 'training', 'val_labels.npy'))

    print(f"Training data shape: {train_patches.shape}")
    print(f"Validation data shape: {val_patches.shape}")

    # Create datasets and dataloaders
    train_dataset = PatchDataset(train_patches, train_labels)
    val_dataset = PatchDataset(val_patches, val_labels)

    batch_size = 16  # Adjust based on available memory
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Initialize model
    model = UNet3D(in_channels=1, out_channels=1, init_features=16)
    model = model.to(device)

    # Initialize loss function and optimizer
    criterion = CombinedLoss(dice_weight=0.5, focal_weight=0.5)
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    # Train model
    model, history = train_model(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        scheduler,
        num_epochs=50,
        patience=10
    )

    # Save model
    os.makedirs(model_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(model_dir, 'model.pth'))

    # Visualize training history
    plot_training_history(history, os.path.join(model_dir, 'training_history.png'))

    print("Model training complete. Saved model to", os.path.join(model_dir, 'model.pth'))

    return model

if __name__ == "__main__":
    # Example usage
    data_dir = '/path/to/preprocessed/data'
    model_dir = '/path/to/save/model'
    main(data_dir, model_dir)
