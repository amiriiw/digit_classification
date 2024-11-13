"""code by amiriiw"""
import os
import numpy as np
import tensorflow as tf
from typing import Tuple
from tensorflow import keras
from sklearn.model_selection import train_test_split


class DataLoader:
    def __init__(self, dataset_dir='dataset', image_size=(28, 28)):
        """Initializes the DataLoader class.
        
        :param dataset_dir: Directory where the dataset is stored. Default is 'dataset'.
        :param image_size: The target size for images. Default is (28, 28).
        """
        self.dataset_dir = dataset_dir
        self.image_size = image_size

    def load_images(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Loads and preprocesses images with different backgrounds from the dataset directory.
        The images are resized and normalized (values between 0 and 1).
        
        :return: The training and testing images and labels, split into training and test sets.
        """
        images, labels = [], []
        for label in range(10):
            folder_path = os.path.join(self.dataset_dir, str(label))
            if not os.path.exists(folder_path):
                print(f"Warning: Folder {folder_path} not found.")
                continue
            for filename in os.listdir(folder_path):
                img_path = os.path.join(folder_path, filename)
                image = keras.preprocessing.image.load_img(img_path, color_mode='grayscale', target_size=self.image_size)
                image_array = np.array(image, dtype=np.float32) / 255.0
                images.append(image_array)
                labels.append(label)
        images = np.array(images).reshape(-1, 28, 28, 1)
        labels = np.array(labels)
        return train_test_split(images, labels, test_size=0.2, random_state=42)

    def augment_data(self, images, labels) -> keras.preprocessing.image.ImageDataGenerator:
        """Applies data augmentation to the images to increase the diversity of the dataset.
        
        :param images: The images to be augmented.
        :param labels: The corresponding labels for the images.
        :return: Augmented image data generator.
        """
        data_gen = keras.preprocessing.image.ImageDataGenerator(
            rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.1,
            zoom_range=0.1,
            fill_mode='nearest'
        )
        return data_gen.flow(images, labels, batch_size=32)


class DigitClassifierModel:
    def __init__(self, input_shape=(28, 28, 1), num_classes=10):
        """Initializes the DigitClassifierModel class, setting up the model architecture.
        
        :param input_shape: The shape of the input images (height, width, channels).
        :param num_classes: The number of output classes (digits 0-9).
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = self._build_model()

    def _build_model(self) -> keras.Sequential:
        """Builds a Convolutional Neural Network (CNN) model for digit classification.
        
        :return: The compiled model ready for training.
        """
        model = keras.Sequential([
            keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Conv2D(64, (3, 3), activation='relu'),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Flatten(),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dense(self.num_classes, activation='softmax')
        ])
        model.compile(
            optimizer='adam',
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
            metrics=['accuracy']
        )
        return model


class DigitClassifier:
    def __init__(self, dataset_dir='dataset'):
        """Initializes the DigitClassifier class.
        
        :param dataset_dir: Directory where the dataset is stored. Default is 'dataset'.
        """
        self.dataset_dir = dataset_dir
        self.image_size = (28, 28)
        self.num_classes = 10
        self.data_loader = DataLoader(dataset_dir=self.dataset_dir, image_size=self.image_size)
        self.model = DigitClassifierModel(input_shape=self.image_size + (1,), num_classes=self.num_classes).model
        self.train_images, self.test_images, self.train_labels, self.test_labels = self._load_data()

    def _load_data(self) -> Tuple[keras.preprocessing.image.ImageDataGenerator, np.ndarray, np.ndarray, np.ndarray]:
        """Loads and preprocesses the data, including applying data augmentation.
        
        :return: Training data generator, test images, and their corresponding labels.
        """
        train_images, test_images, train_labels, test_labels = self.data_loader.load_images()
        train_data_gen = self.data_loader.augment_data(train_images, train_labels)
        return train_data_gen, test_images, train_labels, test_labels

    def train(self, epochs=10) -> None:
        """Trains the model using the augmented data.
        
        :param epochs: The number of epochs to train the model. Default is 10.
        """
        self.model.fit(self.train_images, epochs=epochs, validation_data=(self.test_images, self.test_labels))

    def evaluate(self) -> Tuple[float, float]:
        """Evaluates the model's performance on the test data.
        
        :return: The accuracy and loss of the model on the test data.
        """
        if self.test_images is not None and self.test_labels is not None:
            test_loss, test_acc = self.model.evaluate(self.test_images, self.test_labels)
            print(f'Test accuracy: {test_acc:.4f}')
            print(f'Test loss: {test_loss:.4f}')
            return test_loss, test_acc
        else:
            print("Test data not available.")
            return 0.0, 0.0

    def save_model(self, filename='model.keras') -> None:
        """Saves the trained model to a file.
        
        :param filename: The name of the file to save the model as. Default is 'model.keras'.
        """
        self.model.save(filename)
        print(f'Model saved as {filename}')


if __name__ == "__main__":
    """Creates an instance of the DigitClassifier, trains the model, evaluates it, and saves the model."""
    classifier = DigitClassifier(dataset_dir='dataset')
    classifier.train(epochs=10)
    classifier.evaluate()
    classifier.save_model('digit_classification.keras')

