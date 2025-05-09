#  EuroSAT Image Classification with MLflow Logging

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import BatchNormalization, LeakyReLU, Dropout, GlobalAveragePooling2D, Dense

import mlflow
import mlflow.keras

# Set MLflow experiment
mlflow.set_experiment("EuroSAT Project")

# Dataset Path and Categories
dataset_path = "EuroSAT/2750/"
categories = os.listdir(dataset_path)

# Preprocessing Function
def load_and_preprocess_images(path, size=(128, 128)):
    data, labels = [], []
    for category in categories:
        category_path = os.path.join(path, category)
        for img_name in os.listdir(category_path):
            img = cv2.imread(os.path.join(category_path, img_name))
            img = cv2.resize(img, size)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img / 255.0
            data.append(img)
            labels.append(category)
    return np.array(data, dtype="float32"), np.array(labels)

# Load and preprocess
data, labels = load_and_preprocess_images(dataset_path)
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)
labels_categorical = to_categorical(labels_encoded)

# Dataset Split
X_train, X_temp, y_train, y_temp = train_test_split(data, labels_categorical, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=1/3, random_state=42)

# Data Augmentation
datagen = ImageDataGenerator(rotation_range=20, width_shift_range=0.2,
                             height_shift_range=0.2, horizontal_flip=True, zoom_range=0.2)
val_test_datagen = ImageDataGenerator()
train_generator = datagen.flow(X_train, y_train, batch_size=32, shuffle=True)
val_generator = val_test_datagen.flow(X_val, y_val, batch_size=32, shuffle=False)
test_generator = val_test_datagen.flow(X_test, y_test, batch_size=32, shuffle=False)

# Model Architecture
def build_model():
    model = models.Sequential([
        Input(shape=(128, 128, 3)),

        layers.Conv2D(32, (3, 3), padding='same'),
        BatchNormalization(),
        LeakyReLU(alpha=0.1),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3), padding='same'),
        BatchNormalization(),
        LeakyReLU(alpha=0.1),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(128, (3, 3), padding='same'),
        BatchNormalization(),
        LeakyReLU(alpha=0.1),
        layers.MaxPooling2D((2, 2)),
        Dropout(0.25),

        layers.Conv2D(256, (3, 3), padding='same'),
        BatchNormalization(),
        LeakyReLU(alpha=0.1),
        layers.MaxPooling2D((2, 2)),
        Dropout(0.25),

        layers.Conv2D(512, (3, 3), padding='same'),
        BatchNormalization(),
        LeakyReLU(alpha=0.1),
        layers.MaxPooling2D((2, 2)),
        Dropout(0.25),

        GlobalAveragePooling2D(),
        Dense(256),
        LeakyReLU(alpha=0.1),
        Dropout(0.4),
        Dense(len(categories), activation='softmax')
    ])

    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model

# Build Model
model = build_model()

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)

# MLflow Tracking
with mlflow.start_run(run_name="CNN Model"):
    mlflow.log_param("tensorflow_version", tf.__version__)
    mlflow.log_param("input_shape", "(128, 128, 3)")
    mlflow.log_param("optimizer", "Adam")
    mlflow.log_param("learning_rate", 0.001)
    mlflow.log_param("loss", "categorical_crossentropy")
    mlflow.log_param("augmentation", "rotation, shift, flip, zoom")
    mlflow.log_param("batch_size", 32)
    mlflow.log_param("epochs", 60)
    mlflow.log_param("callbacks", "EarlyStopping, ReduceLROnPlateau")
    
    # Layer configurations
    mlflow.log_param("layers", [
        "Conv2D(32, 3x3), BatchNorm, LeakyReLU, MaxPool",
        "Conv2D(64, 3x3), BatchNorm, LeakyReLU, MaxPool",
        "Conv2D(128, 3x3), BatchNorm, LeakyReLU, MaxPool, Dropout(0.25)",
        "Conv2D(256, 3x3), BatchNorm, LeakyReLU, MaxPool, Dropout(0.25)",
        "Conv2D(512, 3x3), BatchNorm, LeakyReLU, MaxPool, Dropout(0.25)",
        "GlobalAvgPool, Dense(256), LeakyReLU, Dropout(0.4), Output Dense"
    ])

    # Optional: log the model summary as text
    from io import StringIO
    import sys

    stream = StringIO()
    model.summary(print_fn=lambda x: stream.write(x + '\n'))
    summary_string = stream.getvalue()
    stream.close()
    mlflow.log_text(summary_string, "model_summary.txt")
    
    # Train Model
    history = model.fit(
        train_generator,
        epochs=60,
        validation_data=val_generator,
        callbacks=[early_stopping, reduce_lr]
    )

    # Log metrics for each epoch
    for epoch in range(len(history.history["loss"])):
        mlflow.log_metric("train_loss_epoch", history.history["loss"][epoch], step=epoch)
        mlflow.log_metric("val_loss_epoch", history.history["val_loss"][epoch], step=epoch)
        mlflow.log_metric("train_accuracy_epoch", history.history["accuracy"][epoch], step=epoch)
        mlflow.log_metric("val_accuracy_epoch", history.history["val_accuracy"][epoch], step=epoch)


    # Log metrics from last epoch
    final_train_acc = history.history["accuracy"][-1]
    final_val_acc = history.history["val_accuracy"][-1]
    final_train_loss = history.history["loss"][-1]
    final_val_loss = history.history["val_loss"][-1]

    mlflow.log_metric("final_train_accuracy", final_train_acc)
    mlflow.log_metric("final_val_accuracy", final_val_acc)
    mlflow.log_metric("final_train_loss", final_train_loss)
    mlflow.log_metric("final_val_loss", final_val_loss)

    # Log the model
    mlflow.keras.log_model(model, "CNN_Model")
    
    # Call evaluate_model function
    def evaluate_model(model, X_test, y_test, class_names):
        y_pred_probs = model.predict(X_test)
        y_pred = np.argmax(y_pred_probs, axis=1)
        y_true = np.argmax(y_test, axis=1)

        accuracy = accuracy_score(y_true, y_pred) * 100
        cm = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=class_names, yticklabels=class_names)
        plt.xlabel("Predicted Labels")
        plt.ylabel("True Labels")
        plt.title("Confusion Matrix")
        cm_path = os.path.join(mlflow.get_artifact_uri(), "confusion_matrix.png")
        plt.savefig(cm_path)
        mlflow.log_artifact(cm_path)

        report = classification_report(y_true, y_pred, target_names=class_names)
        print(report)
        mlflow.log_text(report, "classification_report.txt")

        print(f"Final Accuracy: {accuracy:.2f}%")
        mlflow.log_metric("final_test_accuracy", accuracy)

    # Evaluate Model and Log Results
    evaluate_model(model, test_generator, y_test, categories)

# Plot training history
import matplotlib.pyplot as plt
import tempfile

plt.figure(figsize=(10, 5))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training and Validation Accuracy')
acc_path = os.path.join(tempfile.gettempdir(), "accuracy_curve.png")
plt.savefig(acc_path)
mlflow.log_artifact(acc_path, artifact_path="plots")
plt.close()

plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')
loss_path = os.path.join(tempfile.gettempdir(), "loss_curve.png")
plt.savefig(loss_path)
mlflow.log_artifact(loss_path, artifact_path="plots")
plt.close()