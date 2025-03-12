from tensorflow.keras.preprocessing import image
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
import os
# Set up paths
train_dir = r"D:\Maran\image_training_data\Food_images_datasets"
batch_size = 124
img_height, img_width = 224, 224

# Get the list of subdirectories (class labels) in the train_dir and directly append to class_labels
class_labels = []
for folder in os.listdir(train_dir):
    folder_path = os.path.join(train_dir, folder)
    if os.path.isdir(folder_path):  # Check if it's a directory
        class_labels.append(folder)  # Append the folder (class label) to class_labels list
num_classes =len(class_labels)
print(class_labels)

# Image data generators for training data augmentation
# train_datagen = ImageDataGenerator(rescale=1./255,
#                                    rotation_range=40,
#                                    width_shift_range=0.2,
#                                    height_shift_range=0.2,
#                                    shear_range=0.2,
#                                    zoom_range=0.2,
#                                    horizontal_flip=True,
#                                    vertical_flip=True,
#                                    fill_mode='nearest'
#                                    )
train_datagen = ImageDataGenerator(rescale=1./255,
                                rotation_range=40,
                                width_shift_range=0.2,
                                height_shift_range=0.2,
                                shear_range=0.2,
                                zoom_range=0.2,
                                horizontal_flip=True,
                                vertical_flip=True,
                                fill_mode='nearest',
                                validation_split=0.2  # Use 20% of the data for validation
                                    )

# Train generator
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'   # for training
)

# Validation generator
validation_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    # class_mode='binary',
    class_mode='categorical',
    subset='validation'  #for validation
)

# Load the pre-trained VGG16 model without the top layer (fully connected layers)
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))

# Freeze layers in the pre-trained model
for layer in base_model.layers:
    layer.trainable = False

# Add custom layers on top of the base model
model = models.Sequential([
    base_model,
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation='softmax')
    # layers.Dense(1, activation='sigmoid')  # Binary classification (Boiled Egg or Green Dal)
])

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Calculate steps per epoch based on your dataset size
steps_per_epoch = train_generator.samples // batch_size

# Train the model
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

history = model.fit(train_generator, epochs=1, steps_per_epoch=steps_per_epoch,
                    validation_data=validation_generator,
                    callbacks=[early_stopping])

# Save the trained model
model.save('food_multi_classifier.h5')

# Load the trained model
model = tf.keras.models.load_model('food_multi_classifier.h5')

# Load a new image for prediction
img_path = r"D:\Maran\image_training_data\Food_images_datasets\Omelets\omelet_13.jpg"  # Adjust image extension if necessary
img = image.load_img(img_path, target_size=(img_height, img_width))
img_array = image.img_to_array(img)  # Convert image to array
img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
img_array /= 255.0  # Normalize the image

# Predict the class
prediction = model.predict(img_array)

# Interpret the prediction
# if prediction[0] > 0.5:
#     print("This image is Green Dal")
# else:
#     print("This image is Boiled Egg")

# Predict the class
predicted_class = np.argmax(prediction, axis=-1)
#
# # Define your class labels (in the same order as your output layer)
# class_labels = ['Green Dal', 'Boiled Egg', 'Fried Rice']  # Add more class names as needed
#
# # Print the predicted class label
print(f"This image is: {class_labels[predicted_class[0]]}")