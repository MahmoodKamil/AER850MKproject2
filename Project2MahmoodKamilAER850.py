# -- coding: utf-8 --
"""
Author: Mahmood Kamil
Student Number: 501061504
AER850 Project 2 
"""

# Part 1: Data Processing
import tensorflow as tf
from keras.preprocessing import image_dataset_from_directory
from keras import layers, models

# Constants
IMG_SIZE = (500, 500)
NUM_CHANNELS = 3
INPUT_SHAPE = (*IMG_SIZE, NUM_CHANNELS)
BATCH_SIZE = 32

# Paths to the dataset directories
train_data_dir = r'C:/Users/mahmo/Downloads/Project 2 Data.zip/Data/train'
valid_data_dir = r'C:/Users/mahmo/Downloads/Project 2 Data.zip/Data/valid'
test_data_dir = r'C:/sers/mahmo/Downloads/Project 2 Data.zip/Data/test'

# Loading datasets
train_ds = image_dataset_from_directory(
    train_data_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='categorical'
)

valid_ds = image_dataset_from_directory(
    valid_data_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='categorical'
)

test_ds = image_dataset_from_directory(
    test_data_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='categorical'
)

# Data augmentation
data_augmentation = models.Sequential([
    layers.Rescaling(1./255),
    layers.RandomZoom(0.1),
    layers.RandomTranslation(0.1, 0.1),
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1)
])

# Apply augmentation to training data
train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y))
valid_ds = valid_ds.map(lambda x, y: (layers.Rescaling(1./255)(x), y))

# Optimize data loading
autotune = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(buffer_size=autotune)
valid_ds = valid_ds.prefetch(buffer_size=autotune)
test_ds = test_ds.prefetch(buffer_size=autotune)

# Inspect batch shape
for images, labels in train_ds.take(1):
    print("Sample Batch Shape:", images.shape, labels.shape)

# Part 2: Neural Network Architecture Design
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=INPUT_SHAPE),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(3, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Part 3: Experimenting with Hyperparameters
from tensorflow.keras.layers import LeakyReLU, ELU

# Alternative model with different activations
alt_model = Sequential([
    Conv2D(32, (3, 3), activation=None, input_shape=INPUT_SHAPE),
    LeakyReLU(alpha=0.1),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation=None),
    LeakyReLU(alpha=0.1),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation=None),
    LeakyReLU(alpha=0.1),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation=None),
    ELU(alpha=1.0),
    Dropout(0.5),
    Dense(3, activation='softmax')
])

alt_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
alt_model.summary()

# Training the model
history = alt_model.fit(train_ds, validation_data=valid_ds, epochs=10)

# Evaluate on the test set
test_loss, test_acc = alt_model.evaluate(test_ds)
print(f"Test Loss: {test_loss}, Test Accuracy: {test_acc}")

# Part 4: Model Evaluation
import matplotlib.pyplot as plt

# Plot training history
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title("Model Accuracy")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title("Model Loss")
plt.legend()

plt.tight_layout()
plt.show()

# Save the model
alt_model.save(r'C:/Users/mahmo/Downloads/Project 2 Model.keras')