## Reading a CSV file
import csv
import numpy as np
import cv2
from tqdm import tqdm 

## /home/workspace
#local_path="./Training_data/data/data/"
local_path = "/opt/carnd_p3/data/"
local_path_ss = "/home/workspace/CarND-Behavioral-Cloning-P3/scene_data/"
local_path_csv = local_path + "driving_log.csv"
local_path_images= local_path + "IMG/"
ignorerowone=1

lines= []

## Reading project CSV file
with open(local_path_csv) as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        if(ignorerowone):
            ignorerowone=0
        else:
            lines.append(line)
images= []
measurements = []

## Processing for project data
for line in tqdm(lines):
    for i in range(3):
        source_path = line[i]
        tokens= source_path.split("/")
        filename = tokens[-1]
        local_path_image = local_path_images + filename    
        image = cv2.cvtColor(cv2.imread(local_path_image),cv2.COLOR_BGR2RGB)
        images.append(image)
    correction=0.2
    measurement = float(line[3])
    measurements.append(measurement)
    measurements.append(measurement+correction)
    measurements.append(measurement-correction)

aug_images= []
aug_measurements = []

for image,measurement in zip(images,measurements):
    aug_images.append(image)
    aug_measurements.append(measurement)
    #flipped_image = cv2.flip(image,1)
    #flipped_measurement = float(measurement) * -1.0
    #aug_images.append(flipped_image)
    #aug_measurements.append(flipped_measurement)    
        
print("----- Read data from CSV file and create Input/Ouput variables ----- ")
print("Length of images array : ",len(images))
print("Length of measurement array: ",len(measurements))

X_train = np.array(aug_images)
Y_train = np.array(aug_measurements)

print("Shape of images array", X_train.shape)
print("Shape of measurement array", Y_train.shape)

## Create a Model in Keras
print("----- Create a Model using Tensorflow Keras ----- ")
import keras

''' LeNet CNN Architecture
from keras.models import Sequential
from keras.layers import Flatten , Dense, Lambda
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D

model = Sequential()
model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,20),(0,0))))
model.add(Convolution2D(6,5,5,activation='relu'))
model.add(MaxPooling2D())
model.add(Convolution2D(16,5,5,activation='relu'))
model.add(Flatten())
model.add(Dense(120))
model.add(Dense(84))
model.add(Dense(1))
'''

### NVidia Architecture : End to End deep learning
from keras.models import Sequential
from keras.layers.core import Dense, Flatten, Activation, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers import Lambda, Cropping2D

model = Sequential()

model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))           
model.add(Convolution2D(24,5,5,subsample=(2,2)))
model.add(Activation('elu'))
model.add(Dropout(0.2))
model.add(Convolution2D(36,5,5,subsample=(2,2)))
model.add(Activation('elu'))
model.add(Convolution2D(48,5,5,subsample=(2,2)))
model.add(Activation('elu'))
model.add(Convolution2D(64,3,3))
model.add(Activation('elu'))
model.add(Convolution2D(64,3,3))
model.add(Activation('elu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Activation('elu'))
model.add(Dense(50))
model.add(Activation('elu'))
model.add(Dense(10))
model.add(Activation('elu'))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')
model.fit(X_train,Y_train, validation_split=0.2, epochs=5,shuffle=True)

model.save('model.h5')
print("Sequential Model created and saved with file name : model.h5")

## python ./CarND-Behavioral-Cloning-P3/drive.py model.h5