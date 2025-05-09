import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.keras
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import keras_tuner as kt
from tensorflow.keras.applications import VGG16

# -------- Argument parsing --------
parser = argparse.ArgumentParser()
parser.add_argument('--learning_rate', type=float, default=0.0005)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--epochs', type=int, default=20)
args = parser.parse_args()

# -------- MLflow setup --------
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("EuroSAT_Classification")

# -------- Load dataset --------
dataset_path = "2750/"  
img_size = (224, 224)

train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_generator = train_datagen.flow_from_directory(
    dataset_path,
    target_size=img_size,
    batch_size=args.batch_size,
    class_mode='categorical',
    subset='training'
)

val_generator = val_datagen.flow_from_directory(
    dataset_path,
    target_size=img_size,
    batch_size=args.batch_size,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

categories = list(train_generator.class_indices.keys())

# -------- Evaluation function --------
def evaluate_model(model, generator, class_names, name):
    y_true = generator.classes
    y_pred_probs = model.predict(generator)
    y_pred = np.argmax(y_pred_probs, axis=1)

    accuracy = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"{name} Confusion Matrix")
    plt.savefig(f"{name}_confusion_matrix.png")
    plt.close()

    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_artifact(f"{name}_confusion_matrix.png")

    print(f"{name} Accuracy: {accuracy * 100:.2f}%")
    print(classification_report(y_true, y_pred, target_names=class_names))

# -------- CNN Model --------
with mlflow.start_run(run_name="CNN_Model"):
    mlflow.log_param("model_type", "CNN")
    mlflow.log_param("learning_rate", args.learning_rate)
    mlflow.log_param("batch_size", args.batch_size)
    mlflow.log_param("epochs", args.epochs)

    cnn_model = models.Sequential([
        layers.Input(shape=(224, 224, 3)),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.4),
        layers.Dense(len(categories), activation='softmax')
    ])

    cnn_model.compile(optimizer=Adam(learning_rate=args.learning_rate),
                      loss='categorical_crossentropy', metrics=['accuracy'])

    cnn_model.fit(train_generator, epochs=args.epochs,
                  validation_data=val_generator,
                  callbacks=[EarlyStopping(patience=5)])

    evaluate_model(cnn_model, val_generator, categories, name="CNN")
    mlflow.keras.log_model(cnn_model, "cnn_model")

# -------- VGG16 + KerasTuner --------
def build_vgg16_model(hp):
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(hp.Choice("dense_units", [256, 512, 1024]), activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(hp.Float("dropout_rate", 0.3, 0.6, step=0.1)),
        layers.Dense(len(categories), activation="softmax")
    ])
    model.compile(
        optimizer=Adam(learning_rate=hp.Choice("learning_rate", [0.0001, 0.0005, 0.001])),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

tuner = kt.RandomSearch(
    build_vgg16_model,
    objective="val_accuracy",
    max_trials=5,
    executions_per_trial=1,
    directory="vgg16_tuner_dir",
    project_name="vgg16_tuning"
)

tuner.search(train_generator, validation_data=val_generator, epochs=5, callbacks=[EarlyStopping(patience=3)])
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

with mlflow.start_run(run_name="VGG16_Tuned"):
    mlflow.log_param("model_type", "VGG16")
    mlflow.log_param("dense_units", best_hps.get("dense_units"))
    mlflow.log_param("dropout_rate", best_hps.get("dropout_rate"))
    mlflow.log_param("learning_rate", best_hps.get("learning_rate"))

    best_model = tuner.hypermodel.build(best_hps)
    best_model.fit(train_generator, epochs=args.epochs,
                   validation_data=val_generator,
                   callbacks=[EarlyStopping(patience=5)])

    evaluate_model(best_model, val_generator, categories, name="VGG16")
    mlflow.keras.log_model(best_model, "vgg16_model")
