import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class SimpleCNN(nn.Module):
    """A simple CNN for binary image classification."""

    def __init__(self, input_channels=3, num_classes=1):
        super(SimpleCNN, self).__init__()
        # Convolutional Layer 1
        self.conv1 = nn.Conv2d(
            in_channels=input_channels,
            out_channels=16,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        # BatchNorm helps stabilize training
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # Output: 16 x 32 x 32

        # Convolutional Layer 2
        self.conv2 = nn.Conv2d(
            in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1
        )
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # Output: 32 x 16 x 16

        # Convolutional Layer 3
        self.conv3 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1
        )
        self.bn3 = nn.BatchNorm2d(64)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)  # Output: 64 x 8 x 8

        # Flatten the output for the fully connected layers
        # Calculate the flattened size: 64 channels * 8 width * 8 height
        self.flattened_size = 64 * 8 * 8

        # Fully Connected Layer 1
        self.fc1 = nn.Linear(self.flattened_size, 128)
        self.dropout = nn.Dropout(0.5)  # Dropout for regularization

        # Fully Connected Layer 2 (Output Layer)
        # Output size is num_classes (1 for binary classification with BCEWithLogitsLoss)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # Conv block 1
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        # Conv block 2
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        # Conv block 3
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))

        # Flatten the output
        x = x.view(-1, self.flattened_size)

        # FC layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)  # Output raw logits (BCEWithLogitsLoss expects logits)
        return x


def get_finetuned_resnet18(num_classes=1, use_pretrained=True):
    """Loads a ResNet18 model, pre-trained on ImageNet, 
       and replaces the final layer for fine-tuning."""
    # Load ResNet18 with specified pre-trained weights
    weights = models.ResNet18_Weights.DEFAULT if use_pretrained else None
    model = models.resnet18(weights=weights)

    # Get the number of input features for the classifier layer
    num_ftrs = model.fc.in_features

    # Replace the final fully connected layer
    # The new layer has num_ftrs inputs and num_classes outputs (1 for our binary case)
    model.fc = nn.Linear(num_ftrs, num_classes)

    return model


# --- Test the model architecture (optional) ---
if __name__ == "__main__":
    print("Testing SimpleCNN architecture...")
    # Create a dummy input tensor matching the expected input shape
    # (batch_size, channels, height, width)
    dummy_input_cnn = torch.randn(4, 3, 64, 64)
    model_cnn = SimpleCNN()
    print(model_cnn)

    # Pass the dummy input through the model
    output_cnn = model_cnn(dummy_input_cnn)
    print(f"\nSimpleCNN Input shape: {dummy_input_cnn.shape}")
    print(f"SimpleCNN Output shape: {output_cnn.shape}")
    # Expected output shape: [batch_size, num_classes] = [4, 1]
    assert output_cnn.shape == (4, 1), f"SimpleCNN Output shape mismatch: {output_cnn.shape}"
    print("SimpleCNN test passed!")

    print("\nTesting Fine-tuned ResNet18 architecture...")
    # ResNet typically expects 224x224, but works with other sizes.
    # Performance might vary with 64x64 inputs.
    dummy_input_resnet = torch.randn(4, 3, 64, 64) 
    model_resnet = get_finetuned_resnet18()
    # print(model_resnet) # Optionally print the full ResNet structure
    
    output_resnet = model_resnet(dummy_input_resnet)
    print(f"ResNet Input shape: {dummy_input_resnet.shape}")
    print(f"ResNet Output shape: {output_resnet.shape}")
    assert output_resnet.shape == (4, 1), f"ResNet Output shape mismatch: {output_resnet.shape}"
    print("Fine-tuned ResNet18 test passed!")
