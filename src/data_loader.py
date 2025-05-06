import os
import glob
import random
from sklearn.model_selection import train_test_split
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import logging

# Setup basic logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# --- Constants ---
IMG_SIZE = 64  # Using 64x64 color images for now
DATA_DIR = "../data"  # Relative path from this script's location
WALDO_PATH_REL = os.path.join(str(IMG_SIZE), "waldo")
NOTWALDO_PATH_REL = os.path.join(str(IMG_SIZE), "notwaldo")
BATCH_SIZE = 64
RANDOM_SEED = 42
VAL_SPLIT = 0.2  # 20% for validation
NUM_WORKERS = 0  # Use 0 for Windows/Jupyter compatibility, can increase later


# --- Dataset Class ---
class WaldoDataset(Dataset):
    """Custom PyTorch Dataset for Waldo image classification."""

    def __init__(self, file_paths, labels, transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        img_path = self.file_paths[idx]
        label = self.labels[idx]

        try:
            # Load image using PIL
            image = Image.open(img_path).convert("RGB")  # Ensure image is RGB
        except Exception as e:
            logging.error(f"Error opening image {img_path}: {e}")
            # Return a placeholder or skip this sample? For now, returning None
            # This should be handled appropriately in the training loop or dataloader collation
            return None, None

        if self.transform:
            image = self.transform(image)

        # Convert label to tensor
        label_tensor = torch.tensor(
            label, dtype=torch.float32
        )  # Use float for BCE loss

        # Add dimension for BCEWithLogitsLoss if using it
        return image, label_tensor.unsqueeze(0)


# --- Transformations ---
# Basic transformations: Convert to Tensor and Normalize
# Normalize using ImageNet stats as a starting point, can be adjusted later
data_transforms = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


# --- Data Loading Function ---
def get_data_loaders(
    data_dir=DATA_DIR,
    waldo_path_rel=WALDO_PATH_REL,
    notwaldo_path_rel=NOTWALDO_PATH_REL,
    batch_size=BATCH_SIZE,
    val_split=VAL_SPLIT,
    random_seed=RANDOM_SEED,
    num_workers=NUM_WORKERS,
    transform=data_transforms,
):
    """Loads data, splits it, and returns DataLoader objects."""

    base_path = os.path.dirname(__file__)  # Get directory of the current script
    abs_data_dir = os.path.abspath(os.path.join(base_path, data_dir))

    waldo_path = os.path.join(abs_data_dir, waldo_path_rel)
    notwaldo_path = os.path.join(abs_data_dir, notwaldo_path_rel)

    logging.info(f"Looking for Waldo images in: {waldo_path}")
    logging.info(f"Looking for non-Waldo images in: {notwaldo_path}")

    # Assuming jpg, adjust glob pattern if needed (e.g., '*.png')
    waldo_files = glob.glob(os.path.join(waldo_path, "*.jpg"))
    notwaldo_files = glob.glob(os.path.join(notwaldo_path, "*.jpg"))

    if not waldo_files:
        logging.warning(f"No Waldo files found in {waldo_path}")
    if not notwaldo_files:
        logging.warning(f"No non-Waldo files found in {notwaldo_path}")
    if not waldo_files and not notwaldo_files:
        logging.error("No image files found. Check DATA_DIR and subfolder paths.")
        return None, None

    logging.info(f"Found {len(waldo_files)} Waldo images.")
    logging.info(f"Found {len(notwaldo_files)} non-Waldo images.")

    # Create lists of file paths and corresponding labels (1 for Waldo, 0 for not Waldo)
    all_files = waldo_files + notwaldo_files
    labels = [1] * len(waldo_files) + [0] * len(notwaldo_files)

    # Combine and shuffle
    combined = list(zip(all_files, labels))
    random.seed(random_seed)
    random.shuffle(combined)
    all_files_shuffled, labels_shuffled = zip(*combined)

    # Split Data into Training and Validation Sets
    if not all_files_shuffled:
        logging.error("No data available for splitting.")
        return None, None

    try:
        train_files, val_files, train_labels, val_labels = train_test_split(
            all_files_shuffled,
            labels_shuffled,
            test_size=val_split,
            random_state=random_seed,
            stratify=labels_shuffled,  # Ensure proportion of classes is similar in train/val
        )
    except ValueError as e:
        logging.error(f"Error during train/test split: {e}. Not enough samples?")
        # Handle cases with very few samples (e.g., only one class represented)
        if len(set(labels_shuffled)) < 2:
            logging.warning(
                "Dataset contains only one class. Cannot stratify. Splitting without stratification."
            )
            train_files, val_files, train_labels, val_labels = train_test_split(
                all_files_shuffled,
                labels_shuffled,
                test_size=val_split,
                random_state=random_seed,
            )
        else:  # Other value errors
            raise e  # Re-raise the error if it's not stratification related

    logging.info(f"Training samples: {len(train_files)}")
    logging.info(f"Validation samples: {len(val_files)}")
    if len(train_labels) > 0:
        logging.info(
            f"Train Waldo proportion: {sum(train_labels)/len(train_labels):.2f}"
        )
    if len(val_labels) > 0:
        logging.info(
            f"Validation Waldo proportion: {sum(val_labels)/len(val_labels):.2f}"
        )

    # Create Datasets
    train_dataset = WaldoDataset(train_files, train_labels, transform=transform)
    val_dataset = WaldoDataset(val_files, val_labels, transform=transform)

    # Create DataLoaders
    # Define a collate function to handle potential None values from dataset __getitem__
    def collate_fn(batch):
        batch = list(
            filter(lambda x: x[0] is not None, batch)
        )  # Filter out samples where image loading failed
        if not batch:
            return torch.empty(0), torch.empty(
                0
            )  # Return empty tensors if batch is empty
        return torch.utils.data.dataloader.default_collate(batch)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )

    logging.info(
        f"DataLoaders created with batch size: {batch_size}, num_workers: {num_workers}"
    )

    return train_loader, val_loader


# --- Main Execution Block (for testing) ---
if __name__ == "__main__":
    logging.info("Testing data loader...")
    train_loader, val_loader = get_data_loaders()

    if train_loader and val_loader:
        logging.info("Data loaders created successfully.")
        # Example: Iterate over one batch to check
        try:
            train_features, train_labels_batch = next(iter(train_loader))
            logging.info(
                f"First training batch shape: Images {train_features.shape}, Labels {train_labels_batch.shape}"
            )
            val_features, val_labels_batch = next(iter(val_loader))
            logging.info(
                f"First validation batch shape: Images {val_features.shape}, Labels {val_labels_batch.shape}"
            )
        except StopIteration:
            logging.warning("Could not retrieve a batch. Data loader might be empty.")
        except Exception as e:
            logging.error(f"Error retrieving batch: {e}")
    else:
        logging.error("Failed to create data loaders.")
