# Refining Image Classifier for Accurate U.S. Currency Identification

## Overview

Imagine walking into a store with a handful of cash, ready to make a purchase. The cashier quickly identifies each bill and calculates the total amount. But what if this process could be automated using a camera? What if a machine could instantly identify and classify different currency bills? This project explores exactly that—building a deep learning model to classify images of U.S. currency.

Currency classification is a critical task in various domains, including retail, banking, and vending machines. For example, an automated teller machine (ATM) needs to distinguish between different denominations to dispense the correct amount of cash. In this project, we developed a model that can classify U.S. currency into categories like "One Dollar," "Five Dollars," "Ten Dollars," etc., using deep learning techniques.

## Project Workflow

### 1. Importing Necessary Libraries
Before diving into the model development, we first import the necessary libraries. These libraries provide tools for data handling, model building, and performance evaluation.

- **Basic Python Libraries**:
  - `os`: Interacts with the operating system to manage files and directories.
  - `numpy`: Fundamental package for numerical computation, especially with arrays.
  - `pandas`: Handles data manipulation and analysis, particularly tabular data.

- **Progress Tracking**:
  - `tqdm`: Adds progress bars to loops, making it easier to track long-running processes.

- **Computer Vision**:
  - `cv2` (OpenCV): An open-source computer vision library used here for image processing tasks.

- **Data Preprocessing**:
  - `LabelEncoder`: Encodes categorical labels into integers.
  - `to_categorical`: Converts class vectors to binary class matrices.

- **Machine Learning and Deep Learning**:
  - `scikit-learn`: Used for data splitting, preprocessing, and evaluation metrics.
  - `Keras & TensorFlow`: Popular deep learning frameworks for building and training neural networks.

- **Visualization**:
  - `Matplotlib & Seaborn`: Libraries essential for creating visual representations of data and model performance.

- **Model Optimization**:
  - `ReduceLROnPlateau`: A Keras callback that reduces the learning rate when a metric has stopped improving.

### 2. Data Preparation

#### **Directory and Extension Setup**
- **Data Directory**: The directory where image data is stored is specified.
- **Image Extensions**: A list of valid image file extensions (JPEG, JPG, PNG) allowed in the dataset.

#### **Image Validation**
We iterate through directories and files, checking the validity of each image:
- **Image Validation**: Ensures that images are readable and correctly formatted.
- **Handling Errors**: Any corrupted or invalid images are removed to maintain a clean dataset.

#### **Loading and Processing Images**
- **Label Assignment**: Assigns labels to images based on the currency denomination.
- **Resizing Images**: All images are resized to a standard size (224x224 pixels).
- **Shuffling Data**: Data is shuffled to prevent bias and improve generalization during model training.

### 3. Visualizing Data

We visualize a few randomly selected images to ensure the labels are correct and the data looks as expected.

### 4. Normalization and Label Encoding
- **Normalizing Images**: Image pixel values are normalized to a range of [0, 1] to improve training efficiency.
- **Label Encoding**: Categorical labels are converted into numeric values and then into one-hot encoded vectors.

### 5. Splitting the Dataset
The dataset is split into training, validation, and test sets to evaluate the model's performance on unseen data.

### 6. Model Development

#### **Loading a Pre-Trained Model**
- **ResNet50**: We use ResNet50, a popular pre-trained model, as the base for our currency classification model. The final classification layers are excluded to add custom layers suitable for our task.

#### **Freezing Layers**
- The pre-trained layers are frozen to retain the knowledge they have learned from the ImageNet dataset.

### 7. Training the Model
We train the model and monitor its accuracy and loss over epochs. The training process is visualized using plots.

### 8. Model Evaluation
After training, the model is evaluated using the test dataset:
- **Classification Report**: Provides detailed metrics like precision, recall, and F1-score for each class.
- **Confusion Matrix**: Displays the performance of the classification model using a heatmap for better visualization.

### 9. Model Selection
We created six models with different architectures and selected the best-performing model based on accuracy and other evaluation metrics.

## Results
The final model achieved a high level of accuracy in classifying U.S. currency images. The confusion matrix and classification report provide insights into the model's performance across different denominations.



## Conclusion
This project demonstrates the effectiveness of deep learning models in automating the process of currency classification. By leveraging pre-trained models and fine-tuning them for specific tasks, we can achieve high accuracy in real-world applications such as ATMs, vending machines, and retail systems.

