# Teeth Classification using Convolutional Neural Networks (CNN)

This project involves developing a deep learning model to classify teeth images into 7 distinct categories using Convolutional Neural Networks (CNN). The model utilizes data augmentation techniques for preprocessing, followed by training and evaluating the model on a dataset of dental images. The project includes image preprocessing, augmentation, CNN model architecture, and model evaluation with performance metrics such as accuracy and loss, along with a confusion matrix for better insight into the model's predictions.

## Project Structure

Teeth_Classification_Project/ │ ├── Teeth_Dataset/ │ ├── Training/ │ ├── Testing/ │ └── Validation/ │ ├── main.py # Main script for training and evaluation ├── requirements.txt # List of dependencies └── README.md # This file

markdown
Copy
Edit

## Dataset

The dataset used in this project consists of images of teeth, categorized into 7 classes. The images are split into three directories:

- `Training/` - Images used for training the model.
- `Testing/` - Images used for evaluating the model.
- `Validation/` - Images used for validating the model during training.

Each image is resized to `32x32` pixels and loaded in RGB format.

## Requirements

Make sure to install the necessary dependencies using the following:

pip install -r requirements.txt

`requirements.txt` contains:

tensorflow>=2.0 matplotlib seaborn scikit-learn numpy


## Model Overview

The project uses a Convolutional Neural Network (CNN) to classify the teeth images. The model architecture includes:

1. **Conv2D Layer**: For feature extraction.
2. **MaxPooling2D Layer**: For downsampling the features.
3. **Dropout Layer**: To prevent overfitting.
4. **Flatten Layer**: To reshape the output for the fully connected layers.
5. **Dense Layers**: For the final classification output with softmax activation.

## Data Preprocessing

Data augmentation is applied during preprocessing to artificially expand the dataset by performing transformations like:

- Horizontal flipping
- Random rotation
- Random zoom
- Random translation (shifting)
- Random brightness and contrast adjustments

This helps the model generalize better and perform robustly across different input variations.

## Training the Model

1. **Data Augmentation** is applied during training.
2. The model is compiled using the **Adam optimizer** and **Sparse Categorical Crossentropy** loss.
3. The model is trained for 200 epochs.

## Model Evaluation

After training, the model is evaluated on the test dataset to measure its accuracy and loss. Additionally, a **confusion matrix** is generated to provide insights into how well the model is performing across different classes.


### Training the Model

Plotting Training and Validation Accuracy & Loss
After training, the model's performance is plotted to visualize:

Training Accuracy vs Validation Accuracy
Training Loss vs Validation Loss
These plots help in understanding the model's generalization ability.

Confusion Matrix
A confusion matrix is displayed after evaluating the model on the test dataset. This matrix helps identify which classes the model is confusing and where improvements can be made.

### Results
The model's accuracy and loss are printed after evaluation on the test dataset:

Test Accuracy: The percentage of correct predictions made by the model.
Test Loss: The loss value, representing the error between the predicted and true labels.
Example Confusion Matrix
The confusion matrix is plotted using seaborn, which provides a clear visual representation of how well the model distinguishes between classes.

### Future Work
Improving the Model: Try different architectures like ResNet or Inception for better accuracy.
Hyperparameter Tuning: Experiment with different batch sizes, learning rates, and augmentations.
Model Deployment: Deploy the trained model in an application for real-time teeth classification.

### Data before augmentation
![fig1](https://github.com/user-attachments/assets/481c4d3e-0c58-4b5c-acf1-bd10f0f7841d)

### Data after augmentation
![fig2](https://github.com/user-attachments/assets/3d5d1bfc-8d65-42f9-94ac-a7dac465a1bf)

### Accuracy
![acc](https://github.com/user-attachments/assets/bd629bfa-2fc7-44d1-a959-48294c2c8a39)


### Loss
![loss](https://github.com/user-attachments/assets/ab5ca292-ab7d-414f-81fb-c0879920f849)

