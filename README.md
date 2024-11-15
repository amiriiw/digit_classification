# Digit Classification

> A deep learning project for classifying handwritten digits using TensorFlow and storing results in a PostgreSQL database. This project includes scripts for image processing, deep learning model development, and database management.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [File Structure](#file-structure)
- [Dependencies](#dependencies)
- [Environment Variables](#environment-variables)
- [Dataset](#dataset)
- [License](#license)

---

## Features

- **Digit Classification Model**: Uses a Convolutional Neural Network (CNN) to classify handwritten digits (0-9).
- **Result Storage**: Saves classification results (predicted digit and confidence) in a PostgreSQL database.
- **Image Processing and Storage**: Processes images and moves classified images to their respective folders.
- **Data Augmentation**: Applies data augmentation to improve model performance.

---

## Installation

Follow these steps to set up the project:

### Clone the repository:

```bash
git clone https://github.com/amiriiw/digit_classification
cd digit_classification
cd Digit-classification
```

### Create a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows use `venv\Scriptsctivate`
```

### Install dependencies:

```bash
pip3 install -r requirements.txt
```

### Set environment variables:

Configure the `.env` file with your database and model information.

## Usage

### Training the Model

Run `train.py` to train the model, evaluate it, and save it:

```bash
python3 train.py
```

- **Model Evaluation**: The script displays the model's accuracy and loss on test data.
- **Model Saving**: The trained model is saved as a `.keras` file.

### Classifying Images

To classify images, use the `detect.py` script, which can process a single image or an entire directory of images.

#### To classify a directory of images:

```bash
python3 detect.py
```

Enter the path to the image or folder when prompted to start the classification process.

---

## File Structure

```
Digit-classification/
├── detect.py                  # Classifies images and saves to database
├── train.py                   # Trains and saves the model
├── requirements.txt           # Project dependencies
├── .env                       # Environment variables
├── dataset/                   # Dataset folder
└── README.md                  # Project documentation
```

---

## Dependencies

This project requires the following libraries:

- **TensorFlow**: For deep learning model development and training.
- **NumPy**: For data processing and numerical operations.
- **psycopg2**: For connecting to PostgreSQL databases.
- **python-dotenv**: For managing environment variables.

Install dependencies with:

```bash
pip3 install -r requirements.txt
```

---

## Environment Variables

Configure the database connection and model path in the `.env` file:

```
DB_NAME="database name"
DB_USER="username"
DB_PASSWORD="password"
DB_HOST="host address"
DB_PORT="port"
MODEL_PATH="model file path"
```

---

## Dataset

Download the training dataset from the following link and place it in the `dataset` folder:

[Download Dataset](https://drive.google.com/file/d/1-172cX2BuWR_zxNXYp9LfC0GVl-An79l/view?usp=drive_link)

---

## License

This project is licensed under the MIT License. See the LICENSE file for more details.
