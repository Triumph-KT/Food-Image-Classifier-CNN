# Food Image Classification Using Convolutional Neural Networks (CNN)

Developed deep learning models to classify food images into three categories (**Bread**, **Soup**, and **Vegetable-Fruit**) using Convolutional Neural Networks (CNN), with a focus on reducing misclassification through advanced regularization, architecture optimization, and detailed error analysis.

## Project Overview

This project automates the classification of food images to support stock photography platforms in efficiently labeling high-volume uploads. By comparing baseline and enhanced CNN architectures, the project identifies challenges in distinguishing visually similar food categories and proposes solutions to improve classification accuracy and minimize mislabeling in real-world applications.

## Dataset

The **Food Image dataset** contains:
- Separate **Training** and **Testing** folders.
- Three classes:
  - Bread
  - Soup
  - Vegetable-Fruit
- Images resized to **150x150 pixels** for model consistency.

## Objectives

- Preprocess food images with normalization and one-hot encoding.
- Build and compare two CNN architectures to assess performance.
- Reduce overfitting and enhance model generalization through regularization techniques.
- Analyze misclassifications to understand class overlap and inform future improvements.
- Deliver actionable insights for dataset refinement and model scaling.

## Methods

### Data Preprocessing:
- Resized all images to **150x150 pixels**.
- Normalized pixel values to the range **[0, 1]**.
- One-hot encoded class labels for multi-class classification.
- Shuffled datasets to ensure balanced learning across batches.

### Model Development:
Two CNN architectures were designed and evaluated:

- **Model 1 (Baseline CNN)**:
  - 3 convolutional layers with **ReLU activation**.
  - MaxPooling after each convolution.
  - Fully connected dense layer with **100 neurons**.
  - Optimizer: **SGD** (learning rate 0.01, momentum 0.9).
  - Trained with **EarlyStopping** and **ModelCheckpoint** for up to **60 epochs**.
  - Observed overfitting after **5 epochs** with validation accuracy plateauing.

- **Model 2 (Enhanced CNN)**:
  - 4 convolutional layers with **Dropout (0.25)** after each block to mitigate overfitting.
  - Fully connected dense layers with **64 and 32 neurons**.
  - Optimizer: **Adam** (learning rate 0.001).
  - Improved stability and reduced misclassification across epochs.

### Evaluation:
- Monitored training and validation accuracy across epochs.
- Measured overfitting via divergence of validation accuracy and loss trends.
- Applied **confusion matrices** and **classification reports** to identify misclassification patterns.
- Analyzed cross-class errors to inform preprocessing and model refinement strategies.

## Results

- **Test Accuracy**: Approximately **70%**.
- **Validation accuracy** stabilized with Dropout and architectural improvements in Model 2.
- Key misclassification patterns identified:
  - High confusion between **Bread** and **Soup**, attributed to similar contextual features (plates, utensils).
  - Overlap between **Bread** and **Vegetable-Fruit** due to shared color tones.
  - Minimal confusion between **Vegetable-Fruit** and **Soup**.
- Enhanced model architecture and regularization reduced overall misclassifications, improving generalization on unseen test data.

## Business/Scientific Impact

- Automates large-scale image labeling in food photography, reducing manual effort and improving operational efficiency for stock photography platforms.
- Identifies critical failure points in classifying visually similar categories, informing data collection strategies to diversify and balance the dataset.
- Provides scalable architecture designs adaptable to larger multi-class food classification tasks for e-commerce, food delivery apps, and digital recipe platforms.
- Recommends ongoing improvements, including advanced data augmentation, background removal, and architectural deepening to enhance model robustness in production environments.

## Technologies Used

- Python
- TensorFlow (Keras)
- OpenCV
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn

## How to Run

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/food-image-classification-cnn.git
    ```

2. Install required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Launch Jupyter Notebook:
    ```bash
    jupyter notebook
    ```

4. Open and run the notebook to:
   - Load and preprocess the dataset.
   - Train both CNN models.
   - Evaluate model performance.
   - Visualize misclassifications and analyze confusion matrices.

## Future Work

- Introduce **Batch Normalization** to further stabilize training.
- Implement **color correction and background removal** to reduce contextual noise in food imagery.
- Apply advanced augmentation techniques (rotations, brightness shifts, cropping) to simulate real-world variability.
- Increase model depth and explore architectures like **ResNet** or **Inception** for improved feature extraction.
- Expand the dataset with more diverse and higher-quality images to mitigate class overlap and enhance generalization.
