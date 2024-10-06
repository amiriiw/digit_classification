# code by amiriiw
import os 
import numpy as np
import mysql.connector  
from tensorflow import keras 
from datetime import datetime  


class DigitPredictor:
    def __init__(self, model_path='digit_classification.keras'):
        """ Load the trained digit classification model. """

        self.model = keras.models.load_model(model_path)
        self.image_size = (28, 28)
        self.db_connection = self.connect_to_database()
        self.create_tables()

    def connect_to_database(self):
        """ Connect to MySQL database. """

        try:
            connection = mysql.connector.connect(
                host='localhost',
                user='user name',
                password='password',
                database='digit_classification'
            )
            print("Database connected successfully.")
            return connection
        except mysql.connector.Error as e:
            print(f"Error connecting to MySQL: {e}")
            return None

    def create_tables(self):
        """ Create a table for storing classified image data if not exists. """

        cursor = self.db_connection.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS classified_images (
                id INT AUTO_INCREMENT PRIMARY KEY,
                digit INT,
                confidence FLOAT,
                timestamp DATETIME,
                folder_name VARCHAR(255),
                file_name VARCHAR(255)
            )
        """)
        self.db_connection.commit()

    def preprocess_image(self, image_path):
        """ Preprocess the image: load, resize, convert to grayscale, normalize. """
        
        image = keras.preprocessing.image.load_img(
            image_path,
            color_mode='grayscale',
            target_size=self.image_size,
            interpolation='bilinear'
        )
        input_image = np.expand_dims(np.array(image, dtype=np.float32) / 255.0, axis=0)
        return input_image

    def predict(self, img_path):
        """ Predict the digit from the preprocessed image. """

        input_image = self.preprocess_image(img_path)
        predictions = self.model.predict(input_image)
        digit = np.argmax(predictions[0])
        confidence = predictions[0][digit]
        print(f'Predicted Digit: {digit}\nConfidence: {confidence:.4f}')
        return digit, confidence

    def save_to_database(self, digit, confidence, folder_name, file_name):
        """ Save classification result (digit, confidence, etc.) into the database. """
        
        cursor = self.db_connection.cursor()
        timestamp = datetime.now()
        digit = int(digit)
        confidence = float(confidence)
        cursor.execute("""
            INSERT INTO classified_images (digit, confidence, timestamp, folder_name, file_name)
            VALUES (%s, %s, %s, %s, %s)
        """, (digit, confidence, timestamp, folder_name, file_name))
        self.db_connection.commit()

    def classify_images(self, input_path):
        """ Classify images in a directory or a single image file. """
        
        if os.path.isdir(input_path):
            for filename in os.listdir(input_path):
                file_path = os.path.join(input_path, filename)
                if file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    digit, confidence = self.predict(file_path)
                    self.save_to_database(digit, confidence, 'result', filename)
                    self.move_image_to_result_folder(file_path, digit)
        elif os.path.isfile(input_path) and input_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            digit, confidence = self.predict(input_path)
            self.save_to_database(digit, confidence, 'result', os.path.basename(input_path))
            self.move_image_to_result_folder(input_path, digit)

    def move_image_to_result_folder(self, image_path, digit):
        """ Move classified image to a folder based on the predicted digit. """

        result_dir = 'result'
        digit_dir = os.path.join(result_dir, str(digit))
        os.makedirs(digit_dir, exist_ok=True)
        new_image_path = os.path.join(digit_dir, os.path.basename(image_path))
        os.rename(image_path, new_image_path)
        print(f'Moved image to: {new_image_path}')


if __name__ == "__main__":
    """ Instantiate the digit predictor and classify images from the input path. """

    predictor = DigitPredictor(model_path='digit_classification.keras')
    test_image_path = input("Enter the image path or directory: ")
    predictor.classify_images(test_image_path)
    predictor.db_connection.close()
