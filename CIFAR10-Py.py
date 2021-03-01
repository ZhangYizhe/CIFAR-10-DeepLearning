import os
from keras.utils import to_categorical

base_dir = "./cifar-10-batches-py"

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='latin1')
    return dict

def pickle(file, obj):
    import pickle
    with open(file, 'wb') as file:
        pickle.dump(obj, file)

train_data = unpickle(os.path.join(base_dir, "train_data"))
train_labels = unpickle(os.path.join(base_dir, "train_labels"))

test_data = unpickle(os.path.join(base_dir, "test_data"))
test_labels = unpickle(os.path.join(base_dir, "test_labels"))

train_data = train_data.astype('float32') / 255
test_data = test_data.astype('float32') / 255

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

from keras import layers
from keras import models
from keras.layers import BatchNormalization

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2))) # extract features
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Dropout(0.2)) # random data to reduce overfitting
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.Dropout(0.5))

model.add(layers.Flatten()) # decrease dimensionality
model.add(BatchNormalization()) # make artificial neural networks faster and more stable through normalization of the input layer by re-centering and re-scaling.
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['acc'])

n_epochs = 50
batch_size = 256
# callbacks_list = None
history = model.fit(
    train_data, train_labels,
    validation_data=(test_data, test_labels),
    epochs=n_epochs,
    batch_size=batch_size,
    # callbacks=callbacks_list
)

# Show figures

import matplotlib.pyplot as plt

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()