import csv
import pdb
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout, Lambda, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

lines = []
with open('./driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

images = []
measurements = []
offset_ang = 0.1
for line in lines:
    i = np.random.randint(3)
    source_path = line[i]
    filename = source_path.split('/')[-1]
    current_path = './IMG/' + filename
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

#pdb.set_trace()

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
model.add(Dropout(0.5))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=6)

model.save('model.h5')
