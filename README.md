# fashion-mnist-cnn

CNN on Fashion-MNIST for the Deep Learning and Transformer Models assignment.

## Model
A convolutional neural network (CNN) built in PyTorch. It has two convolution layers (kernel size 3, padding 1) with ReLU and max pooling, followed by two fully connected layers, and it classifies 28x28 grayscale clothing images into 10 classes. The original model (Model A) uses 16 and 32 filters, and the modified model (Model B) uses 32 and 64 filters.

Tensor shapes for Model A (batch of 128): input (128, 1, 28, 28) -> conv1 (128, 16, 28, 28) -> pool1 (128, 16, 14, 14) -> conv2 (128, 32, 14, 14) -> pool2 (128, 32, 7, 7) -> flattened (128, 1568) -> output logits (128, 10).

## Dataset
Fashion-MNIST, loaded from Hugging Face Datasets: https://huggingface.co/datasets/zalando-datasets/fashion_mnist

The 60,000 training images were split into 54,000 for training and 6,000 for validation. The 10,000 test images were used only for the final evaluation.

## Training
Adam optimizer, learning rate 1e-3, batch size 128, 5 epochs, cross-entropy loss. Both models were trained with the same seed.

## Reproducibility
- Random seed: 42
- Python 3.13.16, PyTorch 2.14.1+cu130, datasets 4.8.5, scikit-learn 1.6.1
- Run on Google Colab (device: cuda)

## Results

Validation set (used to choose the model):

| Model | Filters | Val loss | Val accuracy |
|---|---|---|---|
| A (original) | 16 / 32 | 0.2967 | 0.8918 |
| B (modified) | 32 / 64 | 0.2546 | 0.9058 |

Model B had the higher validation accuracy, so it was selected on validation data.

Held-out test set (10,000 images):

| Model | Test loss | Test accuracy | Macro F1 |
|---|---|---|---|
| A (original) | 0.3151 | 0.8881 | 0.8884 |
| B (modified) | 0.2761 | 0.8991 | 0.8998 |

Model B improved test accuracy by about 1.1 percentage points over Model A (0.8991 vs 0.8881).

Per-class observations (from the classification reports and confusion matrices):
- Trouser was the easiest class for both models (F1 0.9786 for A, 0.9825 for B).
- Shirt was the hardest class for both models (F1 0.7008 for A, 0.7251 for B). It was most often confused with T-shirt/top, Coat and Pullover.
- Pullover recall improved from 0.7190 (A) to 0.8140 (B), while T-shirt/top recall dropped from 0.8700 to 0.8170.

## Files
- `Rosaura_CNN_FashionMNIST.ipynb`: complete notebook with code, tensor-shape checks, training, evaluation and outputs.

## References
- Xiao, H., Rasul, K., & Vollgraf, R. (2017). Fashion-MNIST: a Novel Image Dataset for Benchmarking Machine Learning Algorithms. https://arxiv.org/abs/1708.07747
- PyTorch documentation: https://pytorch.org/docs/stable/index.html
- Code adapted from lecture examples and an AI-assisted draft (Claude), then modified and run by me.
