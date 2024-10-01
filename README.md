# Digit Classification Project
Welcome to the **Digit Classification Project**! This project focuses on training a model to classify handwritten digits and using the trained model to predict digits from new images.

Digit classification/         
├── digit_classification_model_trainer.py
├── digit_classification_detector.py
├── requirements.txt        

## Overview
This project consists of two main components:
1. **digit_classification_model_trainer.py**: This script is responsible for training a model to classify digits based on a dataset of handwritten digits.
2. **digit_classification_detector.py**: This script uses the trained model to predict digits from new images.

## Libraries Used
The following libraries are used in this project:  
- **[tensorflow](https://www.tensorflow.org/)**: TensorFlow is an open-source machine learning library used for training and inference in this project.
- **[numpy](https://numpy.org/)**: NumPy is used for numerical operations on image data.
- **[sklearn](https://scikit-learn.org/)**: Scikit-learn is used for splitting the dataset into training and testing sets.
- **[os](https://python.readthedocs.io/en/stable/library/os.html)**: The OS module is used for directory and file manipulation.
- **[mysql-connector](https://pypi.org/project/mysql-connector-python/)**: This library is used for connecting to a MySQL database to store classification results.

## Detailed Explanation
### `digit_classification_model_trainer.py`
This script is essential for training the digit classification model. The key components of the script are:

- **DigitClassifier Class**: This class manages the entire training process, from loading and preprocessing the dataset to building, training, and saving the model. It uses TensorFlow’s Keras API to create a simple neural network for digit classification.
- **_load_and_preprocess_data() Method**: This method loads the images from the dataset directory, preprocesses them (resizes and normalizes), and splits them into training and testing sets.
- **_build_model() Method**: This method constructs a simple neural network model with a flatten layer, a dense hidden layer, and an output layer for digit classification.
- **train() Method**: This method trains the model on the preprocessed dataset.
- **evaluate() Method**: This method evaluates the model on the test dataset and prints the test accuracy.
- **save_model() Method**: This method saves the trained model to a file for later use.

### `digit_classification_detector.py`
This script uses the trained model to predict digits from new images. The key components of the script are:
- **DigitPredictor Class**: This class loads the trained model and provides methods to preprocess images and make predictions.
- **preprocess_image() Method**: This method loads and preprocesses a new image, preparing it for prediction.
- **predict() Method**: This method predicts the digit in the provided image using the trained model and prints the predicted digit and confidence.
- **save_to_database() Method**: This method saves the prediction results to a MySQL database.
- **classify_images() Method**: This method classifies images from a given directory or a single image.

### How It Works
1. **Model Training**:
    - The `digit_classification_model_trainer.py` script reads images from the specified dataset directory.
    - The images are resized, normalized, and fed into a neural network model, which is trained to classify them into one of ten digit classes (0-9).
    - The trained model is saved for later use.

2. **Digit Prediction**:
    - The `digit_classification_detector.py` script loads the trained model and processes new images.
    - Each image is preprocessed and classified by the model, which outputs the predicted digit and confidence.

## Dataset
The dataset used for training the model can be accessed via this [Dataset](https://drive.google.com/drive/folders/127c2JXxepw8iQ6gRRinPIM5fnRY4ztS9?usp=sharing).

## Installation and Setup
To use this project, follow these steps:

### 1. Clone the repository:
```bash
git clone https://github.com/amiriiw/digit_classification
cd digit_classification
```

### 2. Install the required libraries:
```bash
pip install -r requirements.txt
```

### 3. Setup the database:
You need to create a MySQL database and configure the connection settings.

1. **Install MySQL**: Make sure MySQL is installed and running on your system. You can download it from [here](https://dev.mysql.com/downloads/installer/).
   
2. **Create a new MySQL database**:  
- Open your MySQL shell or MySQL Workbench and run the following commands to create a new database:
```sql
CREATE DATABASE digit_classification;
```

3. **Configure your MySQL connection**:  
- The script `digit_classification_detector.py` is configured to connect to a MySQL database using the following settings:
```python
host='localhost',
user='username',
password='password',
database='digit_classification'
```
- Update these values in the script if your MySQL settings are different (e.g., different username, password, or database name).

4. **Run the prediction script**: The first time you run the script, it will automatically create the necessary tables in the database.
```bash
python digit_classification_detector.py
```

### 4. Prepare the dataset:
Download and prepare your dataset, ensuring it's structured appropriately for training (folders named `0` to `9`, each containing respective digit images).

### 5. Run the model training script:
```bash
python digit_classification_model_trainer.py
```

### 6. Use the trained model to predict digits from new images:
```bash
python digit_classification_detector.py
```

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
