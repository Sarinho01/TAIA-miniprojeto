from tensorflow.keras import layers, models
from data.ship_data_manager import load_and_preprocess_image_expand_dims
from sklearn.metrics import confusion_matrix, f1_score
import numpy as np


def create_ship_model():
    model = models.Sequential()

    model.add(layers.Conv2D(8, (4, 4), strides=2, padding='same', activation='relu', input_shape=(80, 80, 3)))

    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=2))

    model.add(layers.Flatten())
    model.add(layers.Dense(1000, activation='relu'))

    model.add(layers.Dense(2, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


class ShipModel:
    def __init__(self, model_path=None):
        if model_path is None:
            self.model = create_ship_model()
            return

        self.model = models.load_model(model_path)

    def fit(self, X_train, y_train, X_test, y_test):
        self.model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

    def evaluate(self, X_test, y_test):
        return self.model.evaluate(X_test, y_test)

    def save_model(self, name = ""):
        self.model.save(f"result/{name}")

    def predict_image(self, img_path):
        processed_image = load_and_preprocess_image_expand_dims(img_path=img_path,image_size = (80,80))

        prediction =  self.model.predict(processed_image)
        return 0 if prediction[0] > 0.5 else 1

    def calculate_metrics(self, X_test, y_test):
        y_pred = np.argmax(self.model.predict(X_test), axis=-1)  # Para classificação multiclasse

        y_test = np.argmax(y_test, axis=-1)  # Se y_test estiver em one-hot encoded

        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        print(f"tn: {tn}, tp {tp}, fn: {fn}, fp {fp}")

        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)
        ppv = tp / (tp + fp)
        npv = tn / (tn + fn)
        f1 = f1_score(y_test, y_pred)

        return {
            'Sensitivity': sensitivity,
            'Specificity': specificity,
            'Positive Predictive Value (PPV)': ppv,
            'Negative Predictive Value (NPV)': npv,
            'F1 Score': f1
        }

