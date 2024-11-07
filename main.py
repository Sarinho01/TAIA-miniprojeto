from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

from data.ship_data_manager import ShipDataManager
from model.ship_model import ShipModel

all_ship_dir = 'dataset/AllShips'

data_manager = ShipDataManager(all_ship_dir)

images, labels = data_manager.load_images_normalized_from_folder_with_label()
labels = to_categorical(labels, num_classes=2)

X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# ship_model = ShipModel("result/Normal")
ship_model = ShipModel()
ship_model.model.summary()

ship_model.fit(X_train, y_train, X_test, y_test)

test_loss, test_accuracy = ship_model.evaluate(X_test, y_test)
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")

ship_model.save_model("Normal")

metrics = ship_model.calculate_metrics(X_test, y_test)

print(f"Sensitivity: {metrics['Sensitivity']:.4f}")
print(f"Specificity: {metrics['Specificity']:.4f}")
print(f"Positive Predictive Value (PPV): {metrics['Positive Predictive Value (PPV)']:.4f}")
print(f"Negative Predictive Value (NPV): {metrics['Negative Predictive Value (NPV)']:.4f}")
print(f"F1 Score: {metrics['F1 Score']:.4f}")
