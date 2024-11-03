# Feed-Forward Neural Network on Adult Census Income Dataset

## Overview

This project demonstrates a feed-forward neural network implemented using NumPy. The network is trained to predict whether individuals earn over \$50K a year based on census data.

## Dataset

- **Name**: Adult Census Income Dataset
- **Source**: [Kaggle - Adult Census Income Dataset](https://www.kaggle.com/datasets/priyamchoksi/adult-census-income-dataset)
- **Description**: The dataset contains 32,562 instances with 15 attributes, including age, workclass, education, occupation, income, etc. The task is to predict whether a person earns more than \$50K a year.

## Model

- **Architecture**:
  - **Input Layer**: Number of neurons equal to the number of features after preprocessing.
  - **Hidden Layers**: Two hidden layers with 32 neurons each.
  - **Output Layer**: 1 neuron (binary classification).
- **Activation Functions**:
  - **Hidden Layers**: ReLU (Rectified Linear Unit).
  - **Output Layer**: Sigmoid function.
- **Loss Function**: Binary Cross-Entropy Loss (implemented within the backward pass).
- **Optimization**: Gradient Descent with mini-batch training.

## Files

- `NeuralNetwork.py`: Contains the `NeuralNetwork` class with methods for training and prediction.
- `BuildDataset.py`: Provides the `build_dataset` function to load and preprocess the Adult dataset.
- `example.ipynb`: A Jupyter notebook demonstrating how to use the neural network to train on the Adult dataset.
- `README.md`: This file, explaining the project.

## How to Use

1. **Download the Dataset**:

   - Go to the [Kaggle Adult Census Income Dataset page](https://www.kaggle.com/uciml/adult-census-income).
   - Download the `adult.csv` file.
   - The `adult.csv` file is already placed in the project directory.

2. **Clone the Repository**:

   ```bash
   git clone https://github.com/hawraakhalil/feed-forward-neural-network.git
