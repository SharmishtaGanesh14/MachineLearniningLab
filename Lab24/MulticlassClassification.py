import os
import numpy as np
from PIL import Image
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Function to load images from a folder and preprocess them
def load_images_from_folder(base_path, image_size=(32, 32)):
    X = []
    y = []
    labels = {}

    # Get all subdirectories (each representing a class)
    folders = sorted(os.listdir(base_path))

    # Loop over each class (subfolder) inside the base path
    for idx, folder in enumerate(folders):
        labels[folder] = idx  # Map class name to a numeric label
        folder_path = os.path.join(base_path, folder)  # Construct full path to the class folder

        # Loop through all files in the class folder
        for filename in os.listdir(folder_path):
            if filename.endswith('.png') or filename.endswith('.jpg'):
                img_path = os.path.join(folder_path, filename)  # Full image path

                # Load image using PIL
                img = Image.open(img_path)

                # Convert the image to a NumPy array and normalize pixel values to [0, 1]
                img_array = np.array(img) / 255.0

                # Flatten the 3D array (32x32x3) into 1D (3072) for KNN input
                flat_img = img_array.flatten()

                X.append(flat_img)  # Add flattened image to features list
                y.append(idx)  # Add corresponding label (numeric) to labels list

    return np.array(X), np.array(y), labels

# KNN classifier using scikit-learn
def knn_predict_sklearn(X_train, y_train, X_test, k):
    # Train the KNN classifier
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)

    # Predict labels for the test data
    y_pred = knn.predict(X_test)
    return y_pred

def main():
    # Specify the paths to your train and test directories
    train_path = "/home/ibab/Desktop/MachineLearningLab/Lab24/CIFAR-10-images/train"
    test_path = "/home/ibab/Desktop/MachineLearningLab/Lab24/CIFAR-10-images/test"

    # Load train and test data
    X_train, y_train, label_map_train = load_images_from_folder(train_path)
    X_test, y_test, label_map_test = load_images_from_folder(test_path)

    print("Train data shape:", X_train.shape)  # (num_train_images, 3072)
    print("Test data shape:", X_test.shape)  # (num_test_images, 3072)

    # Use k-NN to predict labels for the test data
    y_pred = knn_predict_sklearn(X_train, y_train, X_test, k=5)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")

    # Map predicted labels to class names
    class_names = list(label_map_test.keys())
    predicted_class_names = [class_names[label] for label in y_pred]

    # Print predictions with corresponding class names
    print(f"Predicted labels (numeric): {y_pred[:10]}")  # Display first 10 predictions
    print(f"Predicted labels (class names): {predicted_class_names[:10]}")  # First 10 class names

    # Display the first few test images along with predictions
    for i in range(10):  # Show first 10 test images
        img_array = X_test[i].reshape(32, 32, 3)  # Reshape back to 32x32x3
        img = Image.fromarray((img_array * 255).astype(np.uint8))  # Convert to uint8 and back to image

        plt.imshow(img)
        plt.title(f"Pred: {predicted_class_names[i]}")  # Show the predicted class name
        plt.axis('off')
        plt.show()

if __name__ == "__main__":
    main()
