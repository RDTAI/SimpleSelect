
---

# Simple CNN for CIFAR-10 Classification

This repository contains a simple convolutional neural network (CNN) implemented in PyTorch to classify images from the CIFAR-10 dataset. The model not only predicts the class of the input images but also selectively retains certain outputs based on a learned selection mechanism.

## Requirements

To run this code, you need to have the following libraries installed:

- Python 3.x
- PyTorch
- torchvision

You can install the required libraries using pip:

```bash
pip install torch torchvision
```

## Dataset

This code uses the CIFAR-10 dataset, which contains 60,000 32x32 color images in 10 classes, with 6,000 images per class. The dataset is automatically downloaded if it's not already present.

## Code Overview

1. **Data Loading and Transformation:**
   - The CIFAR-10 dataset is loaded and transformed using normalization.
   - A DataLoader is created for both the training and testing datasets.

2. **Defining the Neural Network:**
   - A simple CNN architecture is defined with two convolutional layers followed by two fully connected layers.
   - The model outputs predictions along with a selection signal for output filtering.

3. **Training Function (`train_model`):**
   - The model is trained for a specified number of epochs.
   - Both the main loss (for the actual class predictions) and a selection loss (for filtered predictions) are calculated and used for optimization.

4. **Testing Function (`test_model`):**
   - The model's performance is evaluated on the test dataset.
   - Accuracy is calculated for both the original model outputs and the selectively retained outputs.

## Usage

To run the training and testing of the model, simply execute the script. You can specify the number of epochs in the `train_model` function:

```python
train_model(num_epochs=10)  # Change 10 to any number of epochs you wish to train
```

### Example

To train the model for 1 epoch and evaluate it, you can use:

```python
train_model(num_epochs=1)
test_model()
```

## Output

The model will print the loss at the end of each training epoch, along with the accuracy of both the original and retained models after testing.

## License

This project is licensed under the MIT License.

---
