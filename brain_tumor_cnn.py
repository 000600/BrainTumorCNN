# Imports
import os
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.layers.experimental.preprocessing import RandomFlip
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.metrics import AUC
from keras.applications.vgg16 import VGG16
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Define classes
class_map = {0 : "No Tumor", 1 : "Tumor"}

# Initialize lists
x = []
y = []

# Get tumor images
tumor_paths = []
for r, d, f in os.walk(r' < PATH TO IMAGES THAT DEPICT A TUMOR > '):
    for fi in f:
        if '.jpg' in fi:
            tumor_paths.append(os.path.join(r, fi)) # Add tumor images to the paths list

# Add images to dataset
for path in tumor_paths:
    img = Image.open(path)
    img = img.resize((128, 128)) # Resize images so that they are easy for the model to understand
    img = np.array(img)
    if (img.shape == (128, 128, 3)):
        x.append(np.array(img)) # Add images to dataset
        y.append(1) # Add corresponding label to Y list

# Get non-tumor images     
nontumor_paths = []
for r, d, f in os.walk(r' < PATH TO IMAGES THAT DO NOT DEPICT A TUMOR > '):
    for fi in f:
        if '.jpg' in fi:
            nontumor_paths.append(os.path.join(r, fi))

# Add images to dataset
for path in nontumor_paths:
    img = Image.open(path)
    img = img.resize((128, 128)) # Resize images so that they are easy for the model to understand
    img = np.array(img)
    if (img.shape == (128, 128, 3)):
        x.append(np.array(img))
        y.append(0) # Append corresponding label to labels list

# Convert dataset into an array
x = np.array(x)

# Convert labels into an array
y = np.array(y)
y = y.reshape(x.shape[0], 1)

# View shapes
print('Dataset Shape:', x.shape)
print('Label Shape:', y.shape)

# Split the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, shuffle = True, random_state = 1)

# Set up epochs and batch size
epochs = 20
batch_size = 32

# Initialize SGD Optimizer
opt = SGD(learning_rate = 0.001)

# Initialize base model (VGG16)
base = VGG16(include_top = False, input_shape = (128, 128, 3))
for layer in base.layers:
  layer.trainable = False # Make VGG16 layers non-trainable so that training goes faster and so that the training process doesn't alter the already tuned values

# Create model
model = Sequential()

# Data augmentation layer and base model
model.add(RandomFlip('horizontal')) # Flip all images along the horizontal axis and add them to the dataset to increase the amount of data the model sees
model.add(base)

# Flatten layer
model.add(Flatten())
model.add(Dropout(0.3))

# Hidden layer
model.add(Dense(256, activation = 'relu'))

# Output layer
model.add(Dense(1, activation = 'sigmoid')) # Sigmoid activation function because the model is a binary classifier

# Configure early stopping
early_stopping = EarlyStopping(min_delta = 0.001, patience = 10, restore_best_weights = True)

# Compile and train model
model.compile(optimizer = opt, loss = 'binary_crossentropy', metrics = ['accuracy', AUC()])
history = model.fit(x_train, y_train, epochs = epochs, batch_size = batch_size, validation_data = (x_test, y_test), callbacks = [early_stopping])

# Visualize  loss and validation loss
history_dict = history.history
loss = history_dict['loss']
val_loss = history_dict['val_loss']
epoch_list = [i for i in range(epochs)]

plt.plot(epoch_list, loss, label = 'Loss')
plt.plot(epoch_list, val_loss, label = 'Validation Loss')
plt.title('Validation and Training Loss Across Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Visualize accuracy and validation accuracy
accuracy = history_dict['accuracy']
val_accuracy = history_dict['val_accuracy']

plt.plot(epoch_list, accuracy, label = 'Training Accuracy')
plt.plot(epoch_list, val_accuracy, label =' Validation Accuracy')
plt.title('Validation and Training Accuracy Across Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Visualize AUC and validation AUC
auc = history_dict['auc']
val_auc = history_dict['val_auc']

plt.plot(epoch_list, auc, label = 'Training AUC')
plt.plot(epoch_list, val_auc, label =' Validation AUC')
plt.title('Validation and Training AUC Across Epochs')
plt.xlabel('Epochs')
plt.ylabel('auc')
plt.legend()
plt.show()

# View model's predictions compared to actual labels
num_viewed_inputs = 10 # Change this number to view more inputs and corresponding predictions and labels

# Get predictions
predictions = model.predict(x_train)

# Display images, predictions, and labels
for i in range(num_viewed_inputs):
  # Get image, prediction, and label
  image = x_test[i]
  pred_prob = float(predictions[i]) # Model's predicted probability that the image is of a certain class
  predicted_class = (0 if pred_prob < 0.5 else 1) # Round the value because the model will predict values in between 0 and 1
  actual_class = y_test[i][0]

  # Get certainty (the probability the model thinks it is correct)
  if predicted_class == 0:
    certainty = (1 - pred_prob) * 100
  else:
    certainty = pred_prob * 100
    
  # View results
  print(f"\nModel's Prediction ({certainty}% certainty): {predicted_class} ({class_map[predicted_class]}) | Actual Class: {actual_class} ({class_map[actual_class]})")

  # Display input image
  fig = plt.figure(figsize = (4, 4))
  plt.axis('off')
  image_display = plt.imshow(image)
  plt.show(image_display)
