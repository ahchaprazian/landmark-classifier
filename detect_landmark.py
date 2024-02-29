import numpy as np
import pandas as pd
import keras
import cv2
from matplotlib import pyplot as plt
import os
import random
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from keras.applications import VGG19
from keras.layers import *
from keras import Sequential
from keras.optimizers import SGD
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import preprocess_input

samples = 20000

# load metadata
df = pd.read_csv("train.csv")
df = df.loc[:samples,:]

# split into unique classes
num_classes = len(df["landmark_id"].unique())
num_data = len(df)
print("Size of training data:", df.shape)
print("Number of unique classes:", num_classes)

# Create a new DataFrame with landmark_id counts
landmark_counts = df['landmark_id'].value_counts().reset_index()
# Rename the columns
landmark_counts.columns = ['landmark_id', 'count']

# Display the first 10 rows
print("Top 10 landmarks:")
print(landmark_counts.head(10))

# Display the last 10 rows
print("\nBottom 10 landmarks:")
print(landmark_counts.tail(10))
print(data['count'].describe())
plt.figure(figsize=(12, 6))  # Set the figure size
bins = np.arange(0, 950, 10)  # Define the bins for the histogram
plt.hist(data['count'], bins=bins, log=True)  # Plot the histogram with a log scale on the y-axis

# Customize the plot appearance
plt.xlabel("Number of images")
plt.ylabel("Occurrences (log scale)")
plt.title("Image Counts per Landmark ID Histogram")
plt.grid(True, which='both', linestyle='--', alpha=0.5)

# Show the plot
plt.show()
# Calculate the number of classes with different datapoint ranges
num_classes_less_than_five = len(data[data['count'] <= 5])
num_classes_between_five_and_fifty = len(data[(data['count'] > 5) & (data['count'] <= 50)])
print("Amount of classes with less than or equal to five datapoints:", num_classes_less_than_five)
print("Amount of classes between five and 50 datapoints:", num_classes_between_five_and_fifty)

# Create bins for the histogram
unique_landmark_ids = np.sort(df["landmark_id"].unique())
bin_edges = np.concatenate((unique_landmark_ids, [unique_landmark_ids[-1] + 1])) - 0.5

# Plot the histogram
plt.figure(figsize=(15, 8))
n, bins, patches = plt.hist(df["landmark_id"], bins=bin_edges)
plt.xlim(0, data['landmark_id'].max())
plt.ylim(0.1, data['count'].max())
plt.yscale('log')
plt.xlabel('Landmark ID')
plt.ylabel('Number of images (log scale)')
plt.title('Histogram of Landmark IDs')
plt.show()
base_path = r"D:\Downloads\google-landmark-master\google-landmark-master\train"
lencoder = LabelEncoder()
lencoder.fit(df["landmark_id"])

#encodes the images and gives it a label
def encode_label(lbl):
    global lencoder
    new_labels = list(set(lbl) - set(lencoder.classes_))
    if new_labels:
        lencoder.classes_ = np.concatenate((lencoder.classes_, new_labels))
    return lencoder.transform(lbl)

# decodes it based on the label
def decode_label(lbl):
    return lencoder.inverse_transform(lbl)

# each langmark has a unique number in the metedata so we get its name from there
# fname describes the file paths that should exist
def get_image_from_number(num):
   fname, label = df.loc[num,:]
   fname = fname + ".jpg"
   path = os.path.join(fname[0], fname[1], fname[2], fname)
   im = cv2.imread(os.path.join(base_path, path))
   return im, label

# pull in four random images fomr our classes and display the landmarks that are shown
def display_random_sample_images():
    print("4 sample images from random classes:")
    fig = plt.figure(figsize=(16, 16))
    for i in range(1, 5):
        random_dir = os.path.join(base_path, *random.choices(os.listdir(base_path), k=3))
        random_img = random.choice(os.listdir(random_dir))
        img = np.array(Image.open(os.path.join(random_dir, random_img)))
        fig.add_subplot(1, 4, i)
        plt.imshow(img)
        plt.axis('off')
    plt.show()
display_random_sample_images()

# Parameters
learning_rate = 0.0001
decay_speed = 1e-6
momentum = 0.09
loss_function = "sparse_categorical_crossentropy"

# Load pre-trained VGG19 model with ImageNet weights
source_model = VGG19(weights='imagenet')

# Add Dropout layers
drop_layer = Dropout(0.5)
drop_layer2 = Dropout(0.5)

# Build the new model
model = Sequential()
# Iterate through the source model layers except the last one and add them to the new model
for layer in source_model.layers[:-1]:  # go through until the last layer
    if layer == source_model.layers[-25]:
        model.add(BatchNormalization())
    model.add(layer)

# Add the final Dense layer with the softmax activation
model.add(Dense(num_classes, activation="softmax"))

# Display model summary
model.summary()

# Compile the model using the RMSprop optimizer
rms = keras.optimizers.RMSprop(learning_rate=learning_rate, momentum=momentum)
model.compile(optimizer=rms, loss=loss_function, metrics=["accuracy"], run_eagerly=True)
print("Model compiled!\n")
def load_image(num, df):
    fname, label = df.iloc[num, :2]
    fname = fname + ".jpg"
    f1, f2, f3 = fname[0], fname[1], fname[2]
    path = os.path.join(f1, f2, f3, fname)
    img = cv2.imread(os.path.join(base_path, path))
    if img is None:
        return None, label
    return img, label

def resize_image(img, target_size):
    if img is None or img.size == 0:
        return None
    return cv2.resize(img, target_size)

def generate_batch(df, start, batch_size):
    images = []
    labels = []
    end = start + batch_size
    if end > len(df):
        end = len(df)
    for idx in range(start, end):
        img, label = load_image(idx, df)
        if img is None:
             continue
        img = resize_image(img, (224, 224))
        if img.size == 0 or img is None:
             continue
        img = img / 255.0
        images.append(img)
        labels.append(label)
    labels = encode_label(labels)
    return np.array(images), np.array(labels)

batch_size = 16
shuffle_each_epoch = True
class_weights = True
num_epochs = 8
train_data, validation_data = np.split(df.sample(frac=1), [int(.8 * len(df))])

print("Training on:", len(train_data), "samples")
print("Validation on:", len(validation_data), "samples")

for epoch in range(num_epochs):
    print("Epoch:", str(epoch + 1) + "/" + str(num_epochs))
    if shuffle_each_epoch:
        train_data = train_data.sample(frac=1)
    for batch in range(int(np.ceil(len(train_data) / batch_size))):
        X_train, y_train = generate_batch(train_data, batch * batch_size, batch_size)
        try:
             model.train_on_batch(X_train, y_train)
        except Exception as e:
            pass

model.save("Trained_Model.h5")

### Test on the training set
batch_size = 16
errors = 0
good_preds = []
bad_preds = []
for it in range(int(np.ceil(len(validate)/batch_size))):
    X_train, y_train = get_batch(validate, it*batch_size, batch_size)
    result = model.predict(X_train)
    cla = np.argmax(result, axis=1)
    for idx, res in enumerate(result):
       #print("Class:", cla[idx], "- Confidence:", np.round(res[cla[idx]],2), "- GT:", y_train[idx])
       if cla[idx] != y_train[idx]:
            errors = errors + 1
            bad_preds.append([batch_size*it + idx, cla[idx], res[cla[idx]]])
       else:
            good_preds.append([batch_size*it + idx, cla[idx], res[cla[idx]]])

print("Errors: ", errors, "Acc:", np.round(100*(len(validate)-errors)/len(validate),2))
# Good predictions
good_preds = np.array(good_preds)
good_preds = np.array(sorted(good_preds, key=lambda x: x[2], reverse=True))
fig = plt.figure(figsize=(16, 16))

if good_preds.size > 0:
    for i in range(1, 6):
        n = int(good_preds[i - 1][0])
        img, lbl = get_image_from_number(n, validate)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        fig.add_subplot(1, 5, i)
        plt.imshow(img)
        lbl2 = np.array(int(good_preds[i - 1][1])).reshape(1, 1)
        sample_cnt = list(df.landmark_id).count(lbl)
        plt.title("Label: " + str(lbl) + "\nClassified as: " + str(decode_label(lbl2)) + "\nSamples in class " + str(lbl) + ": " + str(sample_cnt))
        plt.axis('off')
    plt.show()
else: 
     pass
