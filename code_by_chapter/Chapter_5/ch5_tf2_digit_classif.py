#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: tom verguts
written for TF2

MNIST digit classification; could a standard N-layer network solve this task...?
"""

#%% imports and initializations
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def plot_digits(x, y):
    # plot some images from the data set
    fig, axes = plt.subplots(1, 4, figsize=(7,3))
    for img, label, ax in zip(x, y, axes):
        ax.set_title(label)
        ax.imshow(img)
        ax.axis("off")

def test_performance(model, x_train, x_test, y_train, y_test):
    to_test_x, to_test_y = [x_train, x_test], [y_train, y_test]
    testdata_loss = tf.keras.metrics.CategoricalAccuracy()
    labels =  ["train", "test"]
    print("\n")
    for loop in range(2):
        y_pred = model.predict(to_test_x[loop])
        testdata_loss.update_state(to_test_y[loop], y_pred)
        testdata_loss_summary = np.mean(testdata_loss.result().numpy())*100
        print("mean {} data performance: {:.2f}%".format(labels[loop], testdata_loss_summary))		
	
def preprocess_digits(x_train, y_train,
					        train_size, x_test, y_test, test_size, image_size, n_labels):
    x_train, y_train, x_test, y_test = x_train[:train_size,:], y_train[:train_size], x_test[:test_size,:], y_test[:test_size]
    x_train = x_train.reshape(x_train.shape[0], image_size)  / 255   # from 3D to 2D input data
    x_test  = x_test.reshape(x_test.shape[0], image_size)    / 255   # same here
    y_train = tf.one_hot(y_train, n_labels)
    y_test  = tf.one_hot(y_test, n_labels)
    return x_train, y_train, x_test, y_test	

#%% main code	
if __name__ == "__main__":
	# import digits dataset
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()   
    train_size, test_size = 10000, 50 # downscale to make data set smaller (and training faster)
    learning_rate = 0.0001
    epochs = 10
    batch_size = 100
    batches = int(x_train.shape[0] / batch_size)
    stdev = 0.001
    n_hid = 20

    plot_digits(x_train[:4], y_train[:4])
	  
    # pre-processing
    n_labels = int(np.max(y_train)+1)
    image_size = x_train.shape[1]*x_train.shape[2]
    x_train, y_train, x_test, y_test = preprocess_digits(
		                                  x_train, y_train, train_size, x_test, y_test, test_size, image_size = image_size, n_labels = n_labels)

    # model definition
    model = tf.keras.Sequential([
			tf.keras.Input(shape=(image_size,)),
			tf.keras.layers.Dense(n_hid, activation = "relu"),
			tf.keras.layers.Dense(n_hid, activation = "relu"),
			tf.keras.layers.Dense(n_hid, activation = "relu"),
			tf.keras.layers.Dense(n_labels, activation = "softmax")])
    model.build()

    loss = tf.keras.losses.CategoricalCrossentropy()
    opt = tf.keras.optimizers.Adam(learning_rate = learning_rate)
    model.compile(optimizer = opt, loss = loss)

    # run the model and show a summary after training
    history = model.fit(x_train, y_train, batch_size = batch_size, epochs = epochs)
    model.summary()

    # output
    fig = plt.figure()
    plt.plot(history.history["loss"], color = "black")
    test_performance(model, x_train, x_test, y_train, y_test)




