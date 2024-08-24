# Refining Image Classifier for Accurate U.S. Currency Identification

## Overview

Imagine walking into a store with a handful of cash, ready to make a purchase. The cashier quickly identified each bill and calculated the total amount. But what if this process could be automated using a camera? What if a machine could instantly identify and classify different currency bills? This project explored exactly that—building a deep learning model to classify images of U.S. currency.

Currency classification is a critical task in various domains, including retail, banking, and vending machines. For example, an automated teller machine (ATM) needs to distinguish between different denominations to dispense the correct amount of cash. In this project, a model was developed that could classify U.S. currency into categories like "One Dollar," "Five Dollars," "Ten Dollars," etc., using deep learning techniques.

## Project Workflow

### 1. Importing Necessary Libraries
Before diving into the model development, the necessary libraries were first imported. The code contains tools for data handling, model building, and performance evaluation.

- **Basic Python Libraries**:
  - `os`: Interacted with the operating system to manage files and directories.
  - `numpy`: Provided fundamental tools for numerical computation, especially with arrays.
  - `pandas`: Handled data manipulation and analysis, particularly tabular data.

- **Progress Tracking**:
  - `tqdm`: Added progress bars to loops, making it easier to track long-running processes.

- **Computer Vision**:
  - `cv2` (OpenCV): An open-source computer vision library used for image processing tasks.

- **Data Preprocessing**:
  - `LabelEncoder`: Encoded categorical labels into integers.
  - `to_categorical`: Converted class vectors to binary class matrices.

- **Machine Learning and Deep Learning**:
  - `scikit-learn`: Used for data splitting, preprocessing, and evaluation metrics.
  - `Keras & TensorFlow`: Popular deep learning frameworks used for building and training neural networks.

- **Visualization**:
  - `Matplotlib & Seaborn`: Libraries essential for creating visual representations of data and model performance.

- **Model Optimization**:
  - `ReduceLROnPlateau`: A Keras callback that reduced the learning rate when a metric had stopped improving.

### 2. Data Preparation

#### **Directory and Extension Setup**
- **Data Directory**: The directory where image data was stored was specified.
- **Image Extensions**: A list of valid image file extensions (JPEG, JPG, PNG) allowed in the dataset was created.

#### **Image Validation**
The code iterated through directories and files, checking the validity of each image:
- **Image Validation**: Ensured that images were readable and correctly formatted.
- **Handling Errors**: Any corrupted or invalid images were removed to maintain a clean dataset.

#### **Loading and Processing Images**
- **Label Assignment**: Labels were assigned to images based on the currency denomination.
- **Resizing Images**: All images were resized to a standard size (224x224 pixels).
- **Shuffling Data**: Data was shuffled to prevent bias and improve generalization during model training.

### 3. Visualizing Data

A few randomly selected images were visualized to ensure the labels were correct and the data looked as expected.

### 4. Normalization and Label Encoding
- **Normalizing Images**: Image pixel values were normalized to a range of [0, 1] to improve training efficiency.
- **Label Encoding**: Categorical labels were converted into numeric values and then into one-hot encoded vectors.

### 5. Splitting the Dataset
The dataset was split into training, validation, and test sets to evaluate the model's performance on unseen data.

### 6. Model Development

#### **Loading a Pre-Trained Model**
- **ResNet50**: ResNet50, a popular pre-trained model, was used as the base for the currency classification model. The final classification layers were excluded to add custom layers suitable for the task.

#### **Freezing Layers**
- The pre-trained layers were frozen to retain the knowledge they had learned from the ImageNet dataset.

### 7. Training the Model
The model was trained, and its accuracy and loss were monitored over epochs. The training process was visualized using plots.

### 8. Model Evaluation
After training, the model was evaluated using the test dataset:
- **Classification Report**: Provided detailed metrics like precision, recall, and F1-score for each class.
- **Confusion Matrix**: Displayed the performance of the classification model using a heatmap for better visualization.

### 9. Model Selection
Six models with different architectures were created, and the best-performing model was selected based on accuracy and other evaluation metrics.

## Results
The final model achieved a high level of accuracy in classifying U.S. currency images. The confusion matrix and classification report provided insights into the model's performance across different denominations.

## Conclusion
This project demonstrated the effectiveness of deep learning models in automating the process of currency classification. By leveraging pre-trained models and fine-tuning them for specific tasks, high accuracy was achieved in real-world applications such as ATMs, vending machines, and retail systems.
