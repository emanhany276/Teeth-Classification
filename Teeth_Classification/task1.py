import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

# Image size and batch size
img_size = (32, 32)
batch_size = 32

# Load dataset
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    r"D:\intern\intern\Teeth_Dataset\Training",
    image_size=img_size,
    batch_size=batch_size,
    color_mode="rgb"
)

test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    r"D:\intern\intern\Teeth_Dataset\Testing",
    image_size=img_size,
    batch_size=batch_size,
    color_mode="rgb"
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    r"D:\intern\intern\Teeth_Dataset\Validation",
    image_size=img_size,
    batch_size=batch_size,
    color_mode="rgb"
)

class_names = train_ds.class_names


# ✅ Apply Data Augmentation in Preprocessing
data_augmentation = keras.Sequential([
    keras.layers.RandomFlip("horizontal"),
    keras.layers.RandomRotation(0.1),  # Reduce rotation
    keras.layers.RandomZoom(0.15),  # Reduce zoom
    keras.layers.RandomTranslation(height_factor=0.05, width_factor=0.05),  # Reduce shift
    keras.layers.RandomBrightness(0.15),  
    keras.layers.RandomContrast(0.15),
])



def preprocess(image, label, augment=False):
    #image = tf.cast(image, tf.float32) / 255.0  # Normalize to [0,1]
    if augment:
        image = data_augmentation(image, training=True)  # Ensure augmentation is applied

    image = tf.cast(image, tf.float32) / 255.0  # Normalize to [0,1]    
    return image, label



def train_data_prep(data, shuffle_size, batch_size):
    data = data.map(lambda x, y: preprocess(x, y, augment=True))  # Apply augmentation
    data = data.cache()
    data = data.shuffle(shuffle_size).repeat()
    data = data.prefetch(1)
    return data


def test_data_prep(data, batch_size):
    data = data.map(preprocess)
    data = data.cache()
    data = data.prefetch(1)
    return data


# Apply Preprocessing
train_data_prepared = train_data_prep(train_ds, shuffle_size=1000, batch_size=batch_size)
test_data_prepared = test_data_prep(test_ds, batch_size=batch_size)
val_data_prepared = test_data_prep(val_ds, batch_size=batch_size)



# Function to display images
def display_images(dataset, class_names, num_images=9):
    plt.figure(figsize=(10, 10))
    for images, labels in dataset.take(1):
        for i in range(min(num_images, len(images))):
            ax = plt.subplot(3, 3, i + 1)
            ax.imshow(images[i].numpy().astype("uint8"))
            ax.set_title(class_names[labels[i]])
            ax.axis("off")
    plt.show()

# Display sample images from the training dataset
display_images(train_ds, class_names)
def display_augmented_images(dataset, class_names, num_images=9):
    plt.figure(figsize=(10, 10))
    for images, labels in dataset.take(1):
        for i in range(min(num_images, len(images))):
            ax = plt.subplot(3, 3, i + 1)
            
            # Expand dims to create batch shape (1, 32, 32, 3)
            augmented_image = data_augmentation(tf.expand_dims(images[i], axis=0))
            
            # Remove batch dimension (squeeze) and convert to numpy
            ax.imshow(tf.squeeze(augmented_image).numpy().astype("uint8"))
            ax.set_title(class_names[labels[i]])
            ax.axis("off")
    plt.show()
    
# Display augmented sample images from the training dataset
display_augmented_images(train_ds, class_names)

 
# ✅ CNN Model
input_shape = (32, 32, 3)
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', input_shape=input_shape),
    tf.keras.layers.MaxPooling2D(pool_size=2),
    #tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=2),
    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=2),
    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Dense(units=128, activation='relu'),
    tf.keras.layers.Dense(units=64, activation='relu'),
    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Dense(units=7, activation='softmax'),
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# ✅ Train the model using prepared datasets
history = model.fit(
    train_data_prepared,
    validation_data=val_data_prepared,
    epochs=100,  
    steps_per_epoch=len(train_ds),  # Define how many batches per epoch
    validation_steps=len(val_ds)  
)


# ✅ Plot accuracy and loss
plt.figure(figsize=(12, 5))

# Accuracy plot
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Model Accuracy')
plt.legend()

# Loss plot
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Model Loss')
plt.legend()

plt.show()
