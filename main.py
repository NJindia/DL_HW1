import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from keras.utils.np_utils import to_categorical
from keras import models, layers
# Input Pipeline
mnist = keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()  # Load dataset

# Create network
network = models.Sequential()
network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
network.add(layers.Dense(10, activation='softmax'))
network.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

# Reshape image data
train_images = train_images.reshape((60000, 28 * 28)).astype('float32') / 255
test_images = test_images.reshape((10000, 28 * 28)).astype('float32') / 255

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# Fit and evaluate base model
network.fit(train_images, train_labels, epochs=5, batch_size=128)
test_loss, test_acc = network.evaluate(test_images, test_labels)
print('test_acc:', test_acc)

# Adjusting batch size
network = models.Sequential()
network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
network.add(layers.Dense(10, activation='softmax'))
batches = []
accs = []
adj_losses = []
times = []
for i in range(12):
    b = 2 ** i
    network.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    network.fit(train_images, train_labels, epochs=5, batch_size=b)
    test_loss, test_acc = network.evaluate(test_images, test_labels)
    print(f'batch: {b}\ntest_acc: {test_acc}')
    batches.append(b)
    accs.append(test_acc)
    adj_losses.append(1 - test_loss)
line_a = plt.plot(batches, accs, linestyle='solid', marker='.', label='accuracy')
line_l = plt.plot(batches, adj_losses, linestyle='solid', marker='.', label='adjusted loss (=1-loss) (higher is better)')
plt.legend()
plt.xticks(batches)
plt.title('batch size vs accuracy and loss')
plt.xlabel('batch size')
plt.show()
BATCH_SIZE = 32

# Adjusting epochs
epochs = []
accs = []
losses = []
for epoch in range(1,21):
    optimizer = keras.optimizers.RMSprop()
    network.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    network.fit(train_images, train_labels, epochs=epoch, batch_size=BATCH_SIZE)
    test_loss, test_acc = network.evaluate(test_images, test_labels)
    print(f'epochs: {epoch}\ntest_acc: {test_acc}')
    epochs.append(epoch)
    accs.append(test_acc)
    losses.append(1-test_loss)
line_a = plt.plot(epochs, accs, linestyle='solid', marker='.', label='accuracy')
line_l = plt.plot(epochs, losses, linestyle='solid', marker='.', label='adjusted loss (=1-loss) (higher is better)')
plt.legend()
plt.xticks(epochs)
plt.title('epochs vs accuracy and loss')
plt.xlabel('epochs')
plt.show()
EPOCHS = 4

# Adjusting learning rate, we now know best batch size is 4 or 256, will go with 256 for time sake
lrs = []
accs = []
losses = []
# Smaller is generally better it seems
for lr in [0.0001, 0.00025, 0.0005, 0.00075, 0.001, 0.0025, 0.005, 0.0075, 0.01, 0.025, 0.05, 0.1, 0.5]:
    optimizer = keras.optimizers.RMSprop(learning_rate=lr)
    network.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    network.fit(train_images, train_labels, epochs=EPOCHS, batch_size=BATCH_SIZE)
    test_loss, test_acc = network.evaluate(test_images, test_labels)
    lrs.append(lr)
    accs.append(test_acc)
    losses.append(1-test_loss)
    print(f'learning rate: {lr}\ntest_acc: {test_acc}')
line_a = plt.plot(lrs, accs, linestyle='solid', marker='.', label='accuracy')
line_l = plt.plot(lrs, losses, linestyle='solid', marker='.', label='adjusted loss (=1-loss) (higher is better)')
plt.legend()
plt.title('learning rate vs accuracy and loss')
plt.xlabel('learning rate')
plt.show()
LEARNING_RATE = 0.0001

# Adjusting rho
rhos = []
accs = []
losses = []
for rho in np.linspace(0, 1, 21):
    optimizer = keras.optimizers.RMSprop(learning_rate=LEARNING_RATE, rho=rho)
    network.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    network.fit(train_images, train_labels, epochs=EPOCHS, batch_size=BATCH_SIZE)
    test_loss, test_acc = network.evaluate(test_images, test_labels)
    print(f'rho: {rho}\ntest_acc: {test_acc}')
    rhos.append(rho)
    accs.append(test_acc)
    losses.append(1-test_loss)
line_a = plt.plot(rhos, accs, linestyle='solid', marker='.', label='accuracy')
line_l = plt.plot(rhos, losses, linestyle='solid', marker='.', label='adjusted loss (=1-loss) (higher is better)')
plt.legend()
plt.xticks(rhos)
plt.title('rho vs accuracy and loss')
plt.xlabel('rho')
plt.show()
RHO = 0.95

# Adjusting momentum
moms = []
accs = []
losses = []
for mom in np.linspace(0,1,21):
    optimizer = keras.optimizers.RMSprop(learning_rate=LEARNING_RATE, rho=RHO, momentum=mom)
    network.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    network.fit(train_images, train_labels, epochs=EPOCHS, batch_size=BATCH_SIZE)
    test_loss, test_acc = network.evaluate(test_images, test_labels)
    print(f'momentum: {mom}\ntest_acc: {test_acc}')
    moms.append(mom)
    accs.append(test_acc)
    losses.append(1-test_loss)
line_a = plt.plot(moms, accs, linestyle='solid', marker='.', label='accuracy')
line_l = plt.plot(moms, losses, linestyle='solid', marker='.', label='adjusted loss (=1-loss) (higher is better)')
plt.legend()
plt.xticks(moms)
plt.title('momentum vs accuracy and loss')
plt.xlabel('momentum')
plt.show()
MOMENTUM = 0.05

# Adjusting epsilon
eps = []
accs = []
losses = []
for epsilon in [1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]:
    optimizer = keras.optimizers.RMSprop(learning_rate=LEARNING_RATE, momentum=MOMENTUM, rho=RHO, epsilon=epsilon)
    network.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    network.fit(train_images, train_labels, epochs=EPOCHS, batch_size=BATCH_SIZE)
    test_loss, test_acc = network.evaluate(test_images, test_labels)
    print(f'epsilon: {epsilon}\ntest_acc: {test_acc}')
    eps.append(epsilon)
    accs.append(test_acc)
    losses.append(1-test_loss)
line_a = plt.plot(eps, accs, linestyle='solid', marker='.', label='accuracy')
line_l = plt.plot(eps, losses, linestyle='solid', marker='.', label='adjusted loss (=1-loss) (higher is better)')
plt.legend()
plt.title('epsilon vs accuracy and loss')
plt.xlabel('epsilon')
plt.show()
EPSILON = 0.0001

def _compile(network, optimizer):
    network.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    network.fit(train_images, train_labels, epochs=EPOCHS, batch_size=BATCH_SIZE)
    loss, acc = network.evaluate(test_images, test_labels)
    return acc, 1- loss

opts = ['RMSProp', 'SGD', 'Adam']
accs = []
losses = []
acc, loss = _compile(network, keras.optimizers.RMSprop(learning_rate=LEARNING_RATE, momentum=MOMENTUM, rho=RHO, epsilon=EPSILON))
accs.append(acc)
losses.append(loss)
acc, loss = _compile(network, keras.optimizers.SGD(learning_rate=LEARNING_RATE, momentum=MOMENTUM))
accs.append(acc)
losses.append(loss)
acc, loss = _compile(network, keras.optimizers.Adam(learning_rate=LEARNING_RATE, epsilon=EPSILON))
accs.append(acc)
losses.append(loss)
plt.plot(opts, accs, marker='.', label='accuracy')
plt.plot(opts, losses, marker='.', label='adjusted loss (=1-loss) (higher is better)')
plt.xticks(opts)
plt.legend()
plt.title('Optimizer vs Accuracy and Loss')
plt.show()
