import pandas as pd
import numpy as np
import os
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras


# Adding text for each label
labeling = { 0:'Speed limit (20km/h)',
            1:'Speed limit (30km/h)',
            2:'Speed limit (50km/h)',
            3:'Speed limit (60km/h)',
            4:'Speed limit (70km/h)',
            5:'Speed limit (80km/h)',
            6:'End of speed limit (80km/h)',
            7:'Speed limit (100km/h)',
            8:'Speed limit (120km/h)',
            9:'No passing',
            10:'No passing veh over 3.5 tons',
            11:'Right-of-way at intersection',
            12:'Priority road',
            13:'Yield',
            14:'Stop',
            15:'No vehicles',
            16:'Veh > 3.5 tons prohibited',
            17:'No entry',
            18:'General caution',
            19:'Dangerous curve left',
            20:'Dangerous curve right',
            21:'Double curve',
            22:'Bumpy road',
            23:'Slippery road',
            24:'Road narrows on the right',
            25:'Road work',
            26:'Traffic signals',
            27:'Pedestrians',
            28:'Children crossing',
            29:'Bicycles crossing',
            30:'Beware of ice/snow',
            31:'Wild animals crossing',
            32:'End speed + passing limits',
            33:'Turn right ahead',
            34:'Turn left ahead',
            35:'Ahead only',
            36:'Go straight or right',
            37:'Go straight or left',
            38:'Keep right',
            39:'Keep left',
            40:'Roundabout mandatory',
            41:'End of no passing',
            42:'End no passing veh > 3.5 tons' }

# Parsing data into operable format + picture resize
images = "GTSRB_data/Train/"
data = []
labels = []
for i in range(43):
    img_path = os.path.join(images, str(i))
    for img in os.listdir(img_path):
        im = Image.open(img_path + '/' + img)
        im = im.resize((64, 64))
        im = np.array(im)
        data.append(im)
        labels.append(i)

data = np.array(data)
labels = np.array(labels)

# Data conversion and train test split
x = data.astype('float32')
y = keras.utils.to_categorical(np.array(labels))
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.25, shuffle=True, stratify=y)

# Model implementation
model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=[64, 64, 3]))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Dense(256, activation="relu"))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(256, activation="relu"))
model.add(keras.layers.Dense(43, activation="softmax"))
model.summary()

# Model compilation and fitting
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=20, batch_size=64)

# Making predictions
y_test = np.argmax(y_test,axis=1)
y_pred = np.argmax(model.predict(x_test), axis=1)

print(f'Accuracy score: {accuracy_score(y_test, y_pred)}')

# Show 10 examples of model perfomance on test data
plt.figure(figsize=(25, 25))

# Change start_index value to see new test traffic signs from test set
start_index = 0

for i in range(10):
    plt.subplot(3, 4, i + 1)
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    predicted = y_pred[start_index + i]
    actual = y_test[start_index + i]
    col = 'g'
    if predicted != actual:
        col = 'r'
    plt.xlabel(f'Actual={labeling.get(actual)} || Pred={labeling.get(predicted)}', color=col, fontsize=9)
    plt.imshow(x_test[start_index + i].astype(np.uint8))

plt.show()
6
# Testing block (uncomment if you want test your own image)
# sign_root = input('Please enter root to your sign picture: ')
# #sign_root = 'GTSRB_data/Test/00008.png'
#
# # Get image, resize and convert it to needed format
# temp = []
# test_sign = Image.open(sign_root)
# test_sign = test_sign.resize((64,64))
# test_sign = np.array(test_sign)
# temp.append(test_sign)
# temp = np.array(temp).astype('float32')
#
# # Make a prediction
# pred = np.argmax(model.predict(temp), axis=1)
#
# # Plot test image and its prediction
# plt.imshow(test_sign.astype(np.uint8))
# plt.xlabel(f'Predicted sign name = {labeling.get(pred[0])}', fontsize=14)
# plt.show()