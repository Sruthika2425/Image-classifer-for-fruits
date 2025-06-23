# Image-classifer-for-fruits
# STEP 1: Install Dependencies
!pip install tensorflow matplotlib scikit-learn

# STEP 2: Import Libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# STEP 3: Generate Synthetic Dataset
def generate_fruit_image(color, label, count):
    images = []
    labels = []
    for _ in range(count):
        img = np.ones((100, 100, 3), dtype=np.uint8)
        img[..., :] = color
        images.append(img)
        labels.append(label)
    return images, labels

# Define color-coded fruit images
apple_images, apple_labels = generate_fruit_image([255, 0, 0], 0, 50)      # Red = Apple
banana_images, banana_labels = generate_fruit_image([255, 255, 0], 1, 50)  # Yellow = Banana
orange_images, orange_labels = generate_fruit_image([255, 165, 0], 2, 50)  # Orange = Orange

# Combine and normalize
images = np.array(apple_images + banana_images + orange_images) / 255.0
labels = to_categorical(apple_labels + banana_labels + orange_labels, num_classes=3)

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# STEP 4: Build CNN Model
model = Sequential([
    Conv2D(16, (3, 3), activation='relu', input_shape=(100, 100, 3)),
    MaxPooling2D(2, 2),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(3, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# STEP 5: Train the Model
history = model.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test))

# STEP 6: Visualize Accuracy
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title("Fruit Classification Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.show()

# STEP 7: Predict a Test Sample
sample = np.expand_dims(X_test[0], axis=0)
prediction = model.predict(sample)
class_names = ['Apple', 'Banana', 'Orange']
print("Predicted Class:", class_names[np.argmax(prediction)])
