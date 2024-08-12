"""----------------------------------------------------------------------------
well come, this is amiriiw, this is a simple project about Digit classification.
    in this file we will use the model to detect the digit.
--------------------------------------------------------"""
import numpy as np  # https://numpy.org/devdocs/user/absolute_beginners.html
from tensorflow import keras  # https://www.tensorflow.org/guide/keras
"""----------------------------------------------------------------"""


class DigitPredictor:
    def __init__(self, model_path='digit_classification.keras'):
        self.model = keras.models.load_model(model_path)
        self.image_size = (28, 28)

    def preprocess_image(self, image_path):
        image = keras.preprocessing.image.load_img(
            image_path,
            color_mode='grayscale',
            target_size=self.image_size,
            interpolation='bilinear'
        )
        input_image = np.expand_dims(np.array(image, dtype=np.float32) / 255.0, axis=0)
        return input_image

    def predict(self, img_path):
        input_image = self.preprocess_image(img_path)
        predictions = self.model.predict(input_image)
        digit = np.argmax(predictions[0])
        confidence = predictions[0][digit]
        print(f'Predicted Digit: {digit}\nConfidence: {confidence:.4f}')
        return digit, confidence


if __name__ == "__main__":
    predictor = DigitPredictor(model_path='digit_classification.keras')
    test_image_path = '/content/dataset/5/img_1001.jpg'
    predictor.predict(test_image_path)
"""--------------------------------"""
