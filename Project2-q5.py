import numpy as np
from keras.preprocessing import image
from keras.models import load_model
import matplotlib.pyplot as plt

# Step 5: Model Testing

# Load the trained model
model_path = r"C:/Users/mahmo/Downloads/Project 2 Model.keras"
model = load_model(model_path)

# Define class labels
class_labels = ["Crack", "Missing Head", "Paint Off"]

# Function to predict the class of an image and display it
def predict_and_display(img_path):
    # Load and preprocess the image
    img = image.load_img(img_path, target_size=(500, 500))
    img_array = image.img_to_array(img) / 255.0  # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Get predictions
    predictions = model.predict(img_array, verbose=0)[0]
    predicted_class = class_labels[np.argmax(predictions)]

    # Plot the image with predictions
    plt.imshow(img)
    plt.axis("off")
    prediction_text = "\n".join(
        [f"{class_labels[i]}: {predictions[i]:.2f}" for i in range(len(class_labels))]
    )
    plt.text(
        10,
        450,
        prediction_text,
        color="blue",
        fontsize=14,
        bbox=dict(facecolor="white", alpha=0.7),
    )
    plt.title(f"Prediction: {predicted_class}")
    plt.show()

# Test the function on new images
test_data_dir = r"C:/Users/mahmo/Downloads/Project 2 Data.zip/Data/test"
predict_and_display(f"{test_data_dir}/crack/test_crack.jpg")
predict_and_display(f"{test_data_dir}/missing-head/test_missinghead.jpg")
predict_and_display(f"{test_data_dir}/paint-off/test_paintoff.jpg")