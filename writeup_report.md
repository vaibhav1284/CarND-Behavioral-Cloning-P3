# **Behavioral Cloning** 

The objective of the project is to develop a system and algorithm with which  car drives through a path in a simulator in autonomous mode without explicit programminng for the driving paths(lanes,road conditions etc.). The method employed is deep learning to achieve the project objectives. 

![Simulator Image](./examples/Set2_center_2016_12_01_13_33_05_194.jpg)

The process to develop such a system involves the following steps :  
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode
The below required files are available in the root :
* model.py file - Python source code of the project to train the model
* drive.py - Executes the algorithm in simulator
* model.h5 - Compiled model
* writeup report - Report having processes and steps
* video.mp4 - One complete lap video

#### 2. Submission includes functional code
The car can be driven in the simulator using the compiled model,trained using a deep neural network based on Nvidia End-End deep learning. The car can be driven in the Udacity provided Simulator by executing the below command : 

```
python drive.py model.h5
```

#### 3. Submission code is usable and readable
The code is readable having appropriate sections for specific tasks which are self-explanatory and easy to understand.  The code was developed in python using Udacity project workspace. The model was trained and verified using GPU. 

The below packages were used to develop code : 
```
import csv
import numpy as np
import cv2
from tqdm import tqdm 
import keras
from keras.models import Sequential
from keras.layers.core import Dense, Flatten, Activation, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers import Lambda, Cropping2D
```

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

During the project , two models were tested namely Lenet and Nvidia end-end deep learning. For simple scenarios like straight driving, easy turns LeNet performed well but for difficult scenarios like steep turns (after the bridge) the model failed to predict correct sterring angle values. Therefor,Nvidia model was tried and tested for such scenarios. The results obtained were good.

![Nvidia Architecture](./examples/Nvidia_Architecture.JPG)

The model performed well for the project data but was failing in certain scenarios for which regularization was added.

#### 2. Attempts to reduce overfitting in the model

Regularization was needed to reduce overfitting the model and to perform well for the testing scenarios. Drop out layers were introduced in the model to reduce overfitting. Multiple eperiements were conducted to arrive at the correct dropout layer position and frequency.
Finally, one drop out layer worked well for this project.(model.py line 98)

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 116). I tested the model with epoch sizes in the range from 2-10. The spoch size of 5 worked well.

#### 4. Appropriate training data

Udacity provided training data was used to train the model The provided training data was analysed. It had a total of 24,108 images from center,left and right cameras. The training data had a total of 8 laps : 4 clockwise and 4 anti-colckwise. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The final model architecture was derived at by experimenting and checking different models. Previous results proved that Lenet worked well for identifying patterns in images and hence this model was used a base and to test different scenarios. However, as the scenes were complex , a much deeper and complex model was needed to achieve project objective, Therefore, Nvidia model was used as a base to develop model for this project. 

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. To reduce ovefitting, drop out layer was added in the model. 
Due to good amount and quality of provided training data, additional data was not needed to train the model. 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track. To start with ,**three drop out layers** were added to generalise the model but it was not giving desired results.Eventually, dropout layers were reduced to **one** with a **rate of 0.2**.  

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
