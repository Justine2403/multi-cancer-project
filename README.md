# Cancer Type Classification using Deep Learning
This project is part of Introduction to Deep Learning course (1131_CE6146) at the National Central University.

## Project Overview
This project aims to classify different types of cancer based on medical images using deep learning models. By training and comparing various neural network architectures, the goal is to identify the most effective model for accurate cancer type classification.

The dataset used for this project consists of labeled medical images sourced from Kaggle, organized by cancer type. Multiple models, including Convolutional Neural Networks (CNNs) and Deep Neural Networks (DNNs), are implemented and evaluated based on performance metrics such as validation accuracy and training loss.

The dataset is available at: https://www.kaggle.com/datasets/obulisainaren/multi-cancer

## Dataset and Preprocessing
The dataset consists of medical images categorized by cancer type. The preprocessing pipeline involves:
1. **Resizing** all images to 224x224 pixels.
2. **Normalization** by scaling pixel values to the range [0, 1].
3. **Batching and Shuffling** for efficient training and generalization.
4. **Train-Validation Split** to evaluate model performance on unseen data.
5. **Label Encoding** to convert cancer type labels into numerical values.

The dataset is expected to be located at `data/dataset`. If the dataset is not present, it will be automatically downloaded from Kaggle.

### Kaggle Authentication
To download the dataset from Kaggle, ensure you have a `kaggle.json` API token stored in your home directory or project root. If not, you can generate one from your Kaggle account under "Account" -> "API".

## Model Architectures
The project implements and compares the following architectures:
- **ResNet18** (pre-trained)
- **Convolutional Neural Network (CNN)**
- **Deep Neural Network (DNN)**
- **Support Vector Machine (SVM)**
- **Additional models** can be easily integrated following the provided template.

## Installation

### Prerequisites
- Python 3.7+
- PyTorch (cuda recommended)
- torchvision
- Kaggle API

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/Justine2403/multi-cancer-project
   cd multi-cancer-project
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Place your `kaggle.json` file in the root directory or in `~/.kaggle/`.

## Usage
1. **Download Dataset:**
   The dataset will be automatically downloaded if not found in `data/dataset`.
2. **Train Models:**
   ```bash
   python main.py
   ```
3. **Evaluate and Visualize:**
   The results and model performance will be saved in `results.json`.

## Results
- Model weights and logs are saved after each epoch.
- The model for each architecture is stored in the respective model directory under `models/`.
- Results are visualized using metrics such as training loss, validation accuracy and F1 score.

## How to Contribute
Contributions are welcome! Here's how you can contribute:
1. Fork the repository.
2. Create a new branch for your feature or bug fix:
   ```bash
   git checkout -b feature/new-feature
   ```
3. Make your changes and commit:
   ```bash
   git commit -m "Add new feature"
   ```
4. Push to your branch:
   ```bash
   git push origin feature/new-feature
   ```
5. Submit a pull request.
6. Update the README if your changes affect usage.