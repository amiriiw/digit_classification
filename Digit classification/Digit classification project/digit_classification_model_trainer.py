"""----------------------------------------------------------------------------
well come, this is amiriiw, this is a simple project about Digit classification.
    in this file we will train the dataset to save the model.
----------------------------------------------------------"""
import os  # https://python.readthedocs.io/en/stable/library/os.html
import numpy as np  # https://numpy.org/devdocs/user/absolute_beginners.html
import tensorflow as tf  # https://www.tensorflow.org/
from tensorflow import keras  # https://www.tensorflow.org/guide/keras
from sklearn.model_selection import train_test_split  # https://scikit-learn.org/stable/user_guide.html
"""-------------------------------------------------------------------------------------------------"""


class DigitClassifier:
    def __init__(self, dataset_dir='dataset'):
        self.dataset_dir = dataset_dir
        self.image_size = (28, 28)
        self.num_classes = 10
        self.train_images, self.train_labels, self.test_images, self.test_labels = self._load_and_preprocess_data()
        self.model = self._build_model()

    def _load_and_preprocess_data(self):
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
        self.model.fit(self.train_images, self.train_labels, epochs=epochs)

    def evaluate(self):
        test_loss, test_acc = self.model.evaluate(self.test_images, self.test_labels)
        print(f'Test accuracy: {test_acc:.4f}')

    def save_model(self, filename='model.keras'):
        self.model.save(filename)
        print(f'Model saved as {filename}')


if __name__ == "__main__":
    classifier = DigitClassifier(dataset_dir='dataset')
    classifier.train(epochs=5)
    classifier.evaluate()
    classifier.save_model('digit_classification.keras')
"""-------------------------------------------------"""
