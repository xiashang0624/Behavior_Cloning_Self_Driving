# **Behavioral Cloning for Self-Driving Car**

### Date: 01/08/2018
---

The goals of this project is to teach a car to mimic human driving in a
simulated environment. The training data are collected based on driving images
of three front cameras and steering angle while the car is manually controlled.

The general steps of this project are as follows:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track without leaving the road


[//]: # (Image References)

![](./Video/Test_drive.mp4)
[image1]: ./train_image.png "Training Image"
[image2]: ./recover_image.png "Recover_Image"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network
* writeup_report.md summarizing the results

Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing
```sh
python drive.py model.h5
```
The model.py file contains the code for training and saving the convolution neural network.
The file shows the pipeline I used for training and validating the model.

### Model Architecture and Training Strategy

#### 1.Model architecture

My model is based on the Nvidia published CNN model for self-driving applications.
The final model contains five convolution layers, followed by four fully connected layers.

The model includes RELU layers to introduce nonlinearity , and the data is normalized in the model using a Keras lambda layer.

#### 2. Overfitting
In order to prevent overfitting and improve model performance, randomness has
been introduced in multiple stages during data preprocessing and data
augmentation steps. For example, at each time step, image from one of the three cameras
were randomly selected.

In addition, the model contains a dropout layers to reduce overfitting.

The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. A combination of center lane driving and
recovering from the left and right sides of the road.

### Model Architecture and Training Strategy

#### 1. Solution Approach

The overall strategy for deriving a model architecture was to determine the
steering angle based on the images ahead of the car.

My first step was to use a convolution neural network model similar to the Nvidia CNN model for self-driving applications.
This model is a great starting point for this project because convolutional neural network is good at detecting shapes and features (i.e., curves).
In addition, the proposed network is not too deep (less than 10 layers), which does not
require a lot of computation resources.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set randomly.
A mean squared error is used in the final layer to output the steering angle.

In the first couple of time of running the simulator to see how well the car was driving around track one.
The car kept steering to the left.  Then I augment the dataset by mirrowing the
images (the corresponding steering angle was multiplied by negative 1 as well).

Then I found that the car is difficult to recover from the left side and right side of the
road. Then I started to prepare more training data of recovering from both
sides, as well as making turns.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture consisted of a convolution neural network with the following layers and layer sizes:


| Layers       | Size         | Activation   |
|:------------:|:------------:|:------------:|
|Input|160x320x3||
|Cropping|80x320x3||
|Cov|24x5x5, subsample (2,2)|relu|
|Cov|36x5x5, subsample (2,2)|relu|
|Cov|48x5x5, subsample (2,2)|relu|
|Cov|64x3x3|relu|
|Cov|64x3x3|relu|
|Fully L1|100||
|Fully L2|50||
|Dropout|||
|Fully L3|10||
|Fully L4|1||

#### 3. Training and Validation Sets

To capture good driving behavior, I first recorded two laps on track one using center lane driving.  Here is an frame from the three cameras during driving.
![alt text][image1]
I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to recover from the sides.
![alt text][image2]
The augmented images were randomly shuffled and splitted into training and
validation sets.

An video of the car driving autonumously can be seen in the Youtube vedio.
https://youtu.be/7zVe_XRlSyM

The front camera view of autonomous drive can be seen in the YOutube vedio
here.  https://youtu.be/RsKTHuK-9kM
