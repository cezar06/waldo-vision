import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.ops import nms
import argparse
import os
import logging
from tqdm import tqdm
import time

# Assuming model.py is in the same directory (src)
from model import SimpleCNN, get_finetuned_resnet18

# Re-define transforms or import from data_loader if preferred
from data_loader import data_transforms  # Use the same transforms as training

# Setup basic logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# --- Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WINDOW_SIZE = 64  # Must match the training image size


def load_model(model_path, device):
    """Loads the trained classification model (assumes ResNet18 fine-tuned structure)."""
    try:
        # Instantiate the correct model structure
        model = get_finetuned_resnet18(use_pretrained=False).to(device)
        
        # Load the state dictionary
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()  # Set model to evaluation mode
        logging.info(f"Model loaded successfully from {model_path}")
        return model
    except FileNotFoundError:
        logging.error(f"Model file not found at {model_path}")
        return None
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        return None


def sliding_window(image, step_size, window_size):
    """Slide a window across the image."""
    # slide a window across the image
    for y in range(0, image.shape[0] - window_size + 1, step_size):
        for x in range(0, image.shape[1] - window_size + 1, step_size):
            # yield the current window
            yield (x, y, image[y : y + window_size, x : x + window_size])


def predict_waldo(image_path, model_path, output_dir, stride, threshold, nms_threshold):
    """Performs sliding window detection on an image with NMS."""
    logging.info(f"Starting prediction on {image_path}")
    logging.info(f"Using model: {model_path}, Stride: {stride}, Threshold: {threshold}, NMS IoU: {nms_threshold}")

    # Load the model
    model = load_model(model_path, DEVICE)
    if model is None:
        return

    # Load the image using OpenCV
    try:
        image = cv2.imread(image_path)
        if image is None:
            logging.error(f"Could not read image file: {image_path}")
            return
        # Keep a copy for drawing boxes later
        output_image = image.copy()
        (h, w) = image.shape[:2]
        logging.info(f"Image loaded: {w}x{h} pixels")
    except Exception as e:
        logging.error(f"Error reading image {image_path}: {e}")
        return

    raw_detections = [] # Store raw detections before NMS
    start_time = time.time()

    # Loop over the sliding window locations
    window_count = 0
    # Pre-calculate total windows for tqdm if stride is fixed
    total_windows_y = (h - WINDOW_SIZE + stride) // stride
    total_windows_x = (w - WINDOW_SIZE + stride) // stride
    total_windows = max(0, total_windows_y * total_windows_x)  # Ensure non-negative

    progress_bar = tqdm(
        sliding_window(image, stride, WINDOW_SIZE),
        total=total_windows,
        desc="Sliding Window",
    )

    for x, y, window in progress_bar:
        window_count += 1
        # Ensure window size is correct (edge cases)
        if window.shape[0] != WINDOW_SIZE or window.shape[1] != WINDOW_SIZE:
            continue

        # Preprocess the window (convert to PIL -> apply transforms)
        try:
            window_pil = Image.fromarray(cv2.cvtColor(window, cv2.COLOR_BGR2RGB))
            window_tensor = data_transforms(window_pil)
            # Add batch dimension and send to device
            window_tensor = window_tensor.unsqueeze(0).to(DEVICE)
        except Exception as e:
            logging.warning(f"Error processing window at ({x},{y}): {e}")
            continue

        # Make prediction
        with torch.no_grad():
            output = model(window_tensor)
            # Apply sigmoid to get probability (since using BCEWithLogitsLoss)
            probability = torch.sigmoid(output).item()

        # If probability is above threshold, store detection
        if probability > threshold:
            # Store as (x1, y1, x2, y2, score)
            raw_detections.append([x, y, x + WINDOW_SIZE, y + WINDOW_SIZE, probability])
        # Optional: Update tqdm description with found count
        # progress_bar.set_postfix(found=len(raw_detections))

    end_time = time.time()
    logging.info(
        f"Processed {window_count} windows in {end_time - start_time:.2f} seconds."
    )
    logging.info(
        f"Found {len(raw_detections)} potential detections above threshold {threshold} (before NMS)."
    )
    
    # Apply Non-Maximum Suppression (NMS)
    if not raw_detections:
        logging.info("No raw detections found, skipping NMS.")
        final_detections = []
    else:
        # Convert detections to tensors for NMS function
        boxes = torch.tensor([det[:4] for det in raw_detections], dtype=torch.float32)
        scores = torch.tensor([det[4] for det in raw_detections], dtype=torch.float32)
        
        try:
            # Apply NMS
            keep_indices = nms(boxes, scores, iou_threshold=nms_threshold)
            final_detections = [raw_detections[i] for i in keep_indices]
            logging.info(f"Kept {len(final_detections)} detections after NMS (IoU threshold: {nms_threshold}).")
        except Exception as e:
             logging.error(f"Error during NMS: {e}. Using raw detections.")
             # Fallback to using raw detections if NMS fails for some reason
             final_detections = raw_detections 

    # Draw bounding boxes on the output image using final_detections
    min_score_for_color = threshold
    max_score_for_color = 1.0
    min_intensity = 60 # Darker green for lower scores (above threshold)
    max_intensity = 255 # Bright green for highest scores

    for startX, startY, endX, endY, score in final_detections:
        # Map the score (threshold to 1.0) to the intensity range
        if max_score_for_color > min_score_for_color: # Avoid division by zero if threshold=1.0
            normalized_score = (score - min_score_for_color) / (max_score_for_color - min_score_for_color)
        else:
            normalized_score = 1.0 # Max intensity if threshold is 1.0
        
        green_intensity = int(min_intensity + normalized_score * (max_intensity - min_intensity))
        # Clamp the value just in case
        green_intensity = max(0, min(255, green_intensity)) 
        
        color = (0, green_intensity, 0) # BGR format for OpenCV
        
        cv2.rectangle(
            output_image, (startX, startY), (endX, endY), color, 2
        )  
        text = f"{score:.2f}"  # Display score
        cv2.putText(
            output_image,
            text,
            (startX, startY - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color, # Use the same intensity for text
            2,
        )

    # Save the output image
    os.makedirs(output_dir, exist_ok=True)
    base_filename = os.path.basename(image_path)
    name, ext = os.path.splitext(base_filename)
    output_filename = f"{name}_predictions{ext}"
    output_path = os.path.join(output_dir, output_filename)
    try:
        cv2.imwrite(output_path, output_image)
        logging.info(f"Output image saved to: {output_path}")
    except Exception as e:
        logging.error(f"Error saving output image: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Waldo detection using sliding window classifier."
    )
    parser.add_argument(
        "--image", required=True, type=str, help="Path to the input image file."
    )
    parser.add_argument(
        "--model",
        required=True,
        type=str,
        help="Path to the trained model (.pth file).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="../results",
        help="Directory relative to script to save output images.",
    )
    parser.add_argument(
        "--stride", type=int, default=32, help="Step size for the sliding window."
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5, # Adjusted default based on last run
        help="Probability threshold for classifying a window as Waldo.",
    )
    # Add NMS threshold argument
    parser.add_argument(
        "--nms-threshold",
        type=float,
        default=0.3, # Common default IoU threshold for NMS
        help="IoU threshold for Non-Maximum Suppression.",
    )

    args = parser.parse_args()

    # Construct absolute paths relative to script location if needed
    script_dir = os.path.dirname(__file__)

    # Handle potential relative paths for image and model
    if not os.path.isabs(args.image):
        image_path = os.path.abspath(
            os.path.join(script_dir, "..", args.image)
        )  # Assume image path relative to project root
    else:
        image_path = args.image

    if not os.path.isabs(args.model):
        model_path = os.path.abspath(
            os.path.join(script_dir, "..", args.model)
        )
    else:
        model_path = args.model

    output_dir_abs = os.path.abspath(os.path.join(script_dir, args.output_dir))

    if not os.path.exists(image_path):
        logging.error(f"Input image not found: {image_path}")
    elif not os.path.exists(model_path):
        logging.error(f"Model file not found: {model_path}")
    else:
        predict_waldo(
            image_path, model_path, output_dir_abs, args.stride, args.threshold, args.nms_threshold # Pass NMS threshold
        )
