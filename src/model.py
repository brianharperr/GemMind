import pandas as pd
import numpy as np
from pathlib import Path
import tensorflow as tf
from tensorflow.keras import layers, Model, optimizers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import cv2

class GemPredictor:
    def __init__(self, img_size=256):
        self.img_size = img_size
        self.type_encoder = LabelEncoder()
        self.shape_encoder = LabelEncoder()
        
    def load_and_preprocess_data(self, csv_path):
        # Load CSV data
        df = pd.read_csv(csv_path)
        
        # Convert currency to USD if needed (implement conversion logic here)
        # For now, assuming all prices are USD
        
        # Encode categorical variables
        self.type_encoder.fit(df['type'])
        self.shape_encoder.fit(df['shape'])
        
        # Prepare X (images) and y (multiple targets)
        X = np.zeros((len(df), self.img_size, self.img_size, 3))
        
        # Load and preprocess images
        for idx, img_path in enumerate(df['img']):
            print("../data/" + img_path)
            img = cv2.imread("../data/" + img_path)
            img = cv2.resize(img, (self.img_size, self.img_size))
            img = img / 255.0  # Normalize
            X[idx] = img
            
        y = {
            'price': df['price'].values,
            'width': df['width'].values,
            'length': df['length'].values,
            'depth': df['depth'].values,
            'type': self.type_encoder.transform(df['type']),
            'shape': self.shape_encoder.transform(df['shape'])
        }
        
        return train_test_split(X, y, test_size=0.2, random_state=42)
    
    def build_model(self):
        # Input layer
        input_layer = layers.Input(shape=(self.img_size, self.img_size, 3))
        
        # CNN backbone
        x = layers.Conv2D(32, 3, activation='relu')(input_layer)
        x = layers.MaxPooling2D()(x)
        x = layers.Conv2D(64, 3, activation='relu')(x)
        x = layers.MaxPooling2D()(x)
        x = layers.Conv2D(128, 3, activation='relu')(x)
        x = layers.MaxPooling2D()(x)
        x = layers.Conv2D(256, 3, activation='relu')(x)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        
        # Multiple output heads
        price_output = layers.Dense(1, name='price')(x)
        width_output = layers.Dense(1, name='width')(x)
        length_output = layers.Dense(1, name='length')(x)
        depth_output = layers.Dense(1, name='depth')(x)
        type_output = layers.Dense(len(self.type_encoder.classes_), 
                                 activation='softmax', 
                                 name='type')(x)
        shape_output = layers.Dense(len(self.shape_encoder.classes_), 
                                  activation='softmax', 
                                  name='shape')(x)
        
        # Create model
        model = Model(inputs=input_layer,
                     outputs=[price_output, width_output, length_output,
                             depth_output, type_output, shape_output])
        
        # Compile model
        model.compile(
            optimizer=optimizers.Adam(1e-4),
            loss={
                'price': 'mse',
                'width': 'mse',
                'length': 'mse',
                'depth': 'mse',
                'type': 'sparse_categorical_crossentropy',
                'shape': 'sparse_categorical_crossentropy'
            },
            loss_weights={
                'price': 1.0,
                'width': 1.0,
                'length': 1.0,
                'depth': 1.0,
                'type': 1.0,
                'shape': 1.0
            },
            metrics={
                'price': 'mae',
                'width': 'mae',
                'length': 'mae',
                'depth': 'mae',
                'type': 'accuracy',
                'shape': 'accuracy'
            }
        )
        
        return model
    
    def train(self, csv_path, epochs=50, batch_size=32):
        # Load and preprocess data
        X_train, X_test, y_train, y_test = self.load_and_preprocess_data(csv_path)
        
        # Build and train model
        model = self.build_model()
        
        # Create checkpoint callback
        checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
            'best_opal_model.h5',
            save_best_only=True,
            monitor='val_loss'
        )
        
        # Early stopping callback
        early_stopping_cb = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        # Train the model
        history = model.fit(
            X_train,
            [y_train['price'], y_train['width'], y_train['length'],
             y_train['depth'], y_train['type'], y_train['shape']],
            validation_data=(
                X_test,
                [y_test['price'], y_test['width'], y_test['length'],
                 y_test['depth'], y_test['type'], y_test['shape']]
            ),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[checkpoint_cb, early_stopping_cb]
        )
        
        return model, history
    
    def predict(self, model, image_path):
        # Load and preprocess single image
        img = cv2.imread(image_path)
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = img / 255.0
        img = np.expand_dims(img, axis=0)
        
        # Make predictions
        predictions = model.predict(img)
        
        # Format predictions
        results = {
            'price': predictions[0][0],
            'width': predictions[1][0],
            'length': predictions[2][0],
            'depth': predictions[3][0],
            'type': self.type_encoder.inverse_transform([np.argmax(predictions[4][0])])[0],
            'shape': self.shape_encoder.inverse_transform([np.argmax(predictions[5][0])])[0]
        }
        
        return results

# Example usage
if __name__ == "__main__":
    predictor = GemPredictor()
    model, history = predictor.train('../data/master.csv')
    
    # Example prediction
    results = predictor.predict(model, 'test_opal.jpg')
    print("Predictions:", results)