import csv
import pdb
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout, Lambda, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

# read through the csv file
lines = []
with open('./driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

# prepare training data set by saving the training images and steering angles
images = []
measurements = []
# set an offset angle to correct the left and right image data
offset_ang = 0.1
for line in lines:
    # we first randomly choose which camera to use
    i = np.random.randint(3)
    source_path = line[i]
    filename = source_path.split('/')[-1]
    current_path = './IMG/' + filename
    # convert from BGR to RGB images
    image = cv2.imread(current_path)
    image = image[:,:,::-1]
    images.append(image)
    if i == 0:
        measurement = float(line[3])
    elif i == 1:
        measurement = float(line[3]) + offset_ang
    else:
        measurement = float(line[3]) - offset_ang
    measurements.append(measurement)

# here we augment the training dataset by mirrowing, which created drvining
# images for driving in the counter-clock direction.
aug_images, aug_measurements = [], []
for image, measurement in zip(images, measurements):
    aug_images.append(image)
    aug_measurements.append(measurement)
    aug_images.append(np.fliplr(image))
    aug_measurements.append(measurement * -1.0)


X_train = np.array(aug_images)
y_train = np.array(aug_measurements)

# here save the height, weight and channels of each training image
h, w, c = X_train.shape[1], X_train.shape[2], X_train.shape[3]

# Python debugger, uncomment it for debugging mode
#pdb.set_trace()


# We start with a sequential model using Kerasl.
# The model is based on the Nvidia published CNN structure
# We notice that overfitting is not a big issue here, so the two MaxPooling
# layers are commented out
model = Sequential()
model.add(Lambda(lambda x: x / 125.0 - 1.0, input_shape=(h, w, c)))
model.add(Cropping2D(cropping = ((70, 25),(0,0))))
model.add(Convolution2D(24,5,5, subsample=(2,2),activation = 'relu'))
#model.add(MaxPooling2D())
model.add(Convolution2D(36,5,5, subsample = (2,2),activation = 'relu'))
#model.add(MaxPooling2D())
model.add(Convolution2D(48,5,5, subsample = (2,2),activation = 'relu'))
model.add(Convolution2D(64,3,3,activation = 'relu'))
model.add(Convolution2D(64,3,3,activation = 'relu'))

model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
# Here a Dropout layer is added to prevent over-fitting
model.add(Dropout(0.5))
model.add(Dense(10))
model.add(Dense(1))

# since this is a regression problem: we output the steering angle. We calculate
# the mean square error in the final layer. Adam optimizer is selected so the
# whole training process can be done in less than 10 epoch.
model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=6)

# Save the trained model as 'model.h5'
model.save('model.h5')
