from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam,SGD
#import torch.optim.lr_scheduler as lr_scheduler
from tensorflow.keras.applications import MobileNet
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout,  BatchNormalization, GlobalAveragePooling2D,  Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model


def mobilenet_model(input_shape=(256, 256, 3), num_classes=38):


    base_model = MobileNet(weights='imagenet', include_top=False, input_shape=input_shape)


    base_model.trainable = False

  
    inputs = Input(shape=input_shape)
    x = base_model(inputs, training=False) 
    x = GlobalAveragePooling2D()(x) 
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)  
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x) 
    outputs = Dense(num_classes, activation='softmax')(x) 

    
    model = Model(inputs, outputs)

    return model


# Define the schedule
initial_learning_rate = 1e-5
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=initial_learning_rate,
    decay_steps=10000,  # Number of steps before decay
    decay_rate=0.96,    # Decay factor
    staircase=True      # Apply decay in discrete intervals
)
modelloaded = load_model("mobileNetmodel40,128,1e-5.h5")
# Compile the model with the schedule
modelloaded.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
              loss='categorical_crossentropy',
               metrics=[
                'accuracy',  # Default accuracy
                tf.keras.metrics.Precision(name='precision'),  # Precision metric
                tf.keras.metrics.Recall(name='recall'),        # Recall metric
                tf.keras.metrics.AUC(name='auc'),             # AUC metric
                # tf.keras.metrics.TopKCategoricalAccuracy(k=5, name='top_5_accuracy')  # Top-5 Accuracy
            ])

# Configuration of the image data generators
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,  # Normalization: scales pixel values between 0 and 1
    rotation_range=20,  # Random rotation up to 20 degrees
    width_shift_range=0.2,  # Random horizontal shift up to 20% of the width
    height_shift_range=0.2,  # Random vertical shift up to 20% of the height
    zoom_range=0.2,  # Random zoom up to 20%
    horizontal_flip=True,  # Random horizontal flip
    fill_mode='nearest'  # Filling missing pixels with the nearest value
)

# Generator for the validation or test set (without augmentation)
validation_datagen = ImageDataGenerator(rescale=1.0 / 255.0)  # Only normalization


# Load images from the file system
train_generator = train_datagen.flow_from_directory(
    "/media/riccardo/FDS/New Plant Diseases Dataset(Augmented)/train",  # Path to the training dataset
    target_size=(256, 256),  # Resize images to 224x224
    batch_size=32,  # Number of images per batch
    class_mode='categorical' , # Type of labels: 'categorical' for multi-class classification
    shuffle=True
)

valid_generator = validation_datagen.flow_from_directory(
    "/media/riccardo/FDS/New Plant Diseases Dataset(Augmented)/valid",
    target_size=(256, 256),
    batch_size=32,
    class_mode='categorical',
    shuffle=True
)

history = modelloaded.fit(
    train_generator,  # This could be a flow from ImageDataGenerator or tf.data.Dataset
    validation_data=valid_generator,
    epochs=40,
    initial_epoch=20,
    batch_size=128
)


# Save the final model after training
modelloaded.save('mobileNetmodel40,128,1e-5.h5')

# Optionally, save the history for later analysis
import pickle

with open('mobileNetmodel40,128,1e-5.pkl', 'wb') as file:
    pickle.dump(history.history, file)