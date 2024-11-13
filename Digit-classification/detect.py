"""code by amiriiw"""
import os
import psycopg2
import numpy as np
from typing import Tuple
from psycopg2 import sql
from tensorflow import keras
from datetime import datetime
from dotenv import load_dotenv
from psycopg2.extensions import connection as Psycopg2Connection


load_dotenv()


class DatabaseHandler:
    def __init__(self, db_name, user, password, host='localhost', port='5432'):
        """Initialize the database connection and create necessary tables."""
        self.connection = self.connect(db_name, user, password, host, port)
        self.create_tables()

    def connect(self, db_name: str, user: str, password: str, host: str, port: str) -> Psycopg2Connection:
        """Establishes a connection to the PostgreSQL database.
            
        :param db_name (str): Name of the database.
        :param user (str): Username for the database.
        :param password (str): Password for the database.
        :param host (str): Host address (default is 'localhost').
        :param port (str): Port number (default is '5432').
        """
        
        try:
            connection = psycopg2.connect(
                dbname=db_name,
                user=user,
                password=password,
                host=host,
                port=port
            )
            print("Connected to PostgreSQL database.")
            return connection
        except psycopg2.Error as e:
            print(f"Error connecting to PostgreSQL: {e}")
            return None

    def create_tables(self) -> None:
        """Create a table to store classified images if it doesn't already exist."""
        with self.connection.cursor() as cursor:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS classified_images (
                    id SERIAL PRIMARY KEY,
                    digit INT,
                    confidence FLOAT,
                    timestamp TIMESTAMP,
                    folder_name VARCHAR(255),
                    file_name VARCHAR(255)
                );
            """)
            self.connection.commit()

    def save_classification(self, digit: int, confidence: float, folder_name: str, file_name: str) -> None:
        """Insert classification data into the database.
        
        :param digit (int): The predicted digit.
        :param confidence (float): Confidence level of the prediction.
        :param folder_name (str): Name of the folder where the image is saved.
        :param file_name (str): Name of the image file.
        """
        with self.connection.cursor() as cursor:
            digit = int(digit)        
            confidence = float(confidence)  
            cursor.execute("""
                INSERT INTO classified_images (digit, confidence, timestamp, folder_name, file_name)
                VALUES (%s, %s, %s, %s, %s)
            """, (digit, confidence, datetime.now(), folder_name, file_name))
            self.connection.commit()

    def close_connection(self)-> None:
        """Close the database connection."""
        if self.connection:
            self.connection.close()
            print("Database connection closed.")


class ImageClassifier:
    def __init__(self, model_path, db_handler=None):
        """Load trained model and initialize image classifier."""
        self.model = keras.models.load_model(model_path)
        self.image_size = (28, 28)
        self.db_handler = db_handler

    def preprocess_image(self, image_path: str) -> np.ndarray:
        """Load and preprocess image for prediction.
        
        :param image_path (str): The file path of the image.
        """
        image = keras.preprocessing.image.load_img(
            image_path,
            color_mode='grayscale',
            target_size=self.image_size
        )
        image_array = np.array(image, dtype=np.float32) / 255.0
        return np.expand_dims(image_array, axis=(0, -1))

    def predict_digit(self, image_path: str) -> Tuple[int, float]:
        """Predict digit in the image and return digit and confidence.
        
        :param image_path (str): The file path of the image.
        """
        input_image = self.preprocess_image(image_path)
        predictions = self.model.predict(input_image)
        digit = np.argmax(predictions[0])
        confidence = predictions[0][digit]
        print(f'Predicted Digit: {digit}, Confidence: {confidence:.4f}')
        return digit, confidence

    def classify_image(self, image_path: str, result_folder: str = 'result') -> None:
        """Classify a single image and save the result in the database and move the image.
        
        :param image_path (str): The file path of the image.
        :param result_folder (str): Path to the result folder where the image will be moved.
        """
        digit, confidence = self.predict_digit(image_path)
        file_name = os.path.basename(image_path)
        folder_name = os.path.join(result_folder, str(digit))
        os.makedirs(folder_name, exist_ok=True)
        
        if self.db_handler:
            self.db_handler.save_classification(digit, confidence, folder_name, file_name)

        new_image_path = os.path.join(folder_name, file_name)
        os.rename(image_path, new_image_path)
        print(f'Moved image to: {new_image_path}')

    def classify_images_in_directory(self, directory_path: str, result_folder: str = 'result') -> None:
        """Classify all images in a directory.
        
        :param directory_path (str): Path to the directory containing images.
        :param result_folder (str): Path to the result folder where classified images will be moved.
        """
        for file_name in os.listdir(directory_path):
            file_path = os.path.join(directory_path, file_name)
            if file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                self.classify_image(file_path, result_folder=result_folder)


if __name__ == "__main__":
    """Program start point."""
    db_name = os.getenv('DB_NAME')
    db_user = os.getenv('DB_USER')
    db_password = os.getenv('DB_PASSWORD')
    db_host = os.getenv('DB_HOST')
    db_port = os.getenv('DB_PORT')
    model_path = os.getenv('MODEL_PATH')

    db_handler = DatabaseHandler(db_name=db_name, user=db_user, password=db_password, host=db_host, port=db_port)
    classifier = ImageClassifier(model_path=model_path, db_handler=db_handler)
    
    input_path = input("Enter the image path or directory: ")
    
    if os.path.isdir(input_path):
        classifier.classify_images_in_directory(input_path)
    elif os.path.isfile(input_path) and input_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
        classifier.classify_image(input_path)

    db_handler.close_connection()
