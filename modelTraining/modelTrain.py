import os
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras import optimizers
import keras.losses
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
import matplotlib.pyplot as plt

data = np.load(r"C:\Users\15039\Desktop\School Stuff\MachineLearning\Project\ProjectTest\data\totalData.npy", allow_pickle=True)
X = data[1]
y = data[0]

y = np.atleast_2d(y).T #converting 1D array to 2D column vector
# print(X.shape)
# print(y.shape)

X_train, X_test, y_train, y_test=train_test_split(
    X,y,
    test_size=0.250,
    train_size=0.750,
    random_state=123,
    shuffle=True)
    # stratify=y)


X_train = X_train * (1/256)
X_test = X_train * (1/256)

# Convert the target data into one-hot encoded vectors
y_train = to_categorical(y_train, 4)
y_test = to_categorical(y_test, 4)

model = Sequential()

model.add(Flatten(input_shape=(128,128)))
model.add(Dense(128,activation='elu'))
model.add(Dense(512,activation='elu'))
model.add(Dense(512,activation='elu'))
model.add(Dense(256,activation='elu'))
model.add(Dense(128,activation='elu'))
model.add(Dense(64,activation='elu'))
model.add(Dense(32,activation='elu'))
model.add(Dense(16,activation='elu'))
model.add(Dense(4,activation='softmax'))

opt = optimizers.Adam()

model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

TrainStats = model.fit(
    x=X_train,
    y=y_train,
    batch_size=64,
    epochs=20,
    verbose='auto',
    # callbacks=None,
    validation_split=0.3,
    # validation_data=None,
    shuffle=True,
    # class_weight=None,
    # sample_weight=None,
    # initial_epoch=0,
    # steps_per_epoch=None,
    # validation_steps=None,
    # validation_batch_size=None,
    # validation_freq=1,
    # max_queue_size=10,
    # workers=1,
    # use_multiprocessing=False,
)

model.summary()
os.chdir(r"C:\Users\15039\Desktop\School Stuff\MachineLearning\Project\ProjectTest\modelTraining")
model.save('model2.h5')

print(TrainStats.history.keys())

preds = model.predict(X_test)


plt.plot(TrainStats.history['accuracy'])
plt.plot(TrainStats.history['val_accuracy'])
plt.legend(['train','test'])

