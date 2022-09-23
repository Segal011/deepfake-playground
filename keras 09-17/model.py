import h5py
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras.losses import sparse_categorical_crossentropy
from tensorflow.keras.optimizers import Adam

# Model configuration
batch_size = 100
img_width, img_height, img_num_channels = 28, 28, 1
loss_function = sparse_categorical_crossentropy
no_classes = 10
no_epochs = 1
optimizer = Adam()
validation_split = 0.2
verbosity = 1

# Load MNIST data
f = h5py.File(r'C:\Users\37060\Desktop\Magistrinis-darbas\Projektas\keras\train.hdf5', 'r')
input_train = f['image'][...]
label_train = f['label'][...]
f.close()
f = h5py.File(r'C:\Users\37060\Desktop\Magistrinis-darbas\Projektas\keras\test.hdf5', 'r')
input_test = f['image'][...]
label_test = f['label'][...]
f.close()

# Reshape data
input_train = input_train.reshape((len(input_train), img_width, img_height, img_num_channels))
input_test  = input_test.reshape((len(input_test), img_width, img_height, img_num_channels))

# Determine shape of the data
input_shape = (img_width, img_height, img_num_channels)

# Create the model
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(no_classes, activation='softmax'))

# Display a model summary
model.summary()

# Compile the model
model.compile(loss=loss_function,
              optimizer=optimizer,
              metrics=['accuracy'])

# model.train_on_batch(x_train[:10], y_train[:10])
# kernel, bias = dense.get_weights()

# from numpy.testing import assert_allclose
# assert_allclose(kernel, 1.)
# assert_allclose(bias, 2.)

# Fit data to model
history = model.fit(input_train, label_train,
            batch_size=batch_size,
            epochs=no_epochs,
            verbose=verbosity,
            validation_split=validation_split)

# Generate generalization metrics
score = model.evaluate(input_test, label_test, verbose=0)
print(f'Test loss: {score[0]} / Test accuracy: {score[1]}')

import json

with open("sample.json", "w") as outfile:
    outfile.write(json.dumps(score))