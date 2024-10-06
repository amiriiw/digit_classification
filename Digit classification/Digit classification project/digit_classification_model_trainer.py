# code by amiriiw
import os  
import numpy as np  
import tensorflow as tf  
from tensorflow import keras 
from sklearn.model_selection import train_test_split


class DigitClassifier:
    def __init__(self, dataset_dir='dataset'):
        """ Initialize with the dataset directory and set image size and number of classes. """

        self.dataset_dir = dataset_dir
        self.image_size = (28, 28)
        self.num_classes = 10
        self.train_images, self.train_labels, self.test_images, self.test_labels = self._load_and_preprocess_data()
        self.model = self._build_model()

    def _load_and_preprocess_data(self):
        """ Load images and labels from the dataset directory. """

        images = []
        labels = []
        for label in range(self.num_classes):
            folder_path = os.path.join(self.dataset_dir, str(label))
            for filename in os.listdir(folder_path):
                img_path = os.path.join(folder_path, filename)
                image = keras.preprocessing.image.load_img(img_path, color_mode='grayscale', target_size=self.image_size)
                image_array = np.array(image, dtype=np.float32) / 255.0
                images.append(image_array)
                labels.append(label)
        images = np.array(images)
        labels = np.array(labels)
        train_images, test_images, train_labels, test_labels = train_test_split(
            images, labels, test_size=0.2, random_state=42
        )
        return train_images, train_labels, test_images, test_labels

    def _build_model(self):
        """ Build a simple neural network model for digit classification. """

        model = keras.Sequential([
            keras.layers.Flatten(input_shape=self.image_size + (1,)),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dense(self.num_classes)
        ])
        model.compile(
            optimizer='adam',
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy']
        )
        return model

    def train(self, epochs=10):
        """ Train the model using the training data. """

        self.model.fit(self.train_images, self.train_labels, epochs=epochs)

    def evaluate(self):
        """ Evaluate the model's performance on the test data. """

        test_loss, test_acc = self.model.evaluate(self.test_images, self.test_labels)
        print(f'Test accuracy: {test_acc:.4f}')

    def save_model(self, filename='model.keras'):
        """ Save the trained model to a file. """

        self.model.save(filename)
        print(f'Model saved as {filename}')


if __name__ == "__main__":
    """ Create an instance of the DigitClassifier, train the model, evaluate it, and save the model. """

    classifier = DigitClassifier(dataset_dir='dataset')
    classifier.train(epochs=10)
    classifier.evaluate()
    classifier.save_model('digit_classification.keras')
