#!/usr/bin/env python3

"""
Description: Deep Learning Neural Network (DLNN) approach for gesture recognition
Author: Ayusman Saha
"""
import sys
import numpy as np
import keypoints as kp
import matplotlib.pyplot as plt
from keras import models, layers, utils

K = 0                       # number of folds to process for validation
EPOCHS = 100                # number of epochs to train the model
BATCH_SIZE = 16             # training data batch size
SPLIT = 0.75                # split percentage for training vs. testing data
NORMALIZATION = 'polar'     # type of data normalization

def plot(epochs, loss, acc, val_loss, val_acc):
        fig, ax = plt.subplots(2)

        # plot loss
        plt.subplot(2, 1, 1)  
        plt.plot(epochs, loss, '--b', label="Training")
        plt.plot(epochs, val_loss, '-g', label="Validation")
        plt.title('Model Performance')
        plt.ylabel('Loss')
        plt.grid()
        plt.legend()

        # plot accuracy
        plt.subplot(2, 1, 2)  
        plt.plot(epochs, acc, '--b', label="Training")
        plt.plot(epochs, val_acc, '-g', label="Validation")
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.grid()
        plt.legend()

        plt.show()

def build_model(inputs, outputs, summary=False):
    model = models.Sequential()

    model.add(layers.Dense(16, activation='relu', input_shape=(inputs,)))
    model.add(layers.Dense(16, activation='relu'))
    model.add(layers.Dense(16, activation='relu'))
    model.add(layers.Dense(16, activation='relu'))
    model.add(layers.Dense(outputs, activation='softmax'))
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])

    if summary is True:
        model.summary()

    return model

def k_fold_cross_validation(data, labels, k, epochs=1, batch_size=1):
    size = data.shape[0] // k
    scores = [[], [], [], []]

    for i in range(k):
        print("Processing fold " + str(i + 1) + "/" + str(k))

        # validation data and lables
        val_data = data[size * i:size * (i + 1)]
        val_labels = labels[size * i:size * (i + 1)]

        # training data and labels
        train_data = np.concatenate([data[:size * i], data[size * (i + 1):]], axis=0)
        train_labels = np.concatenate([labels[:size * i], labels[size * (i + 1):]], axis=0)

        # build model
        model = build_model(data.shape[1], labels.shape[1])

        # train model
        history = model.fit(train_data,
                            train_labels,
                            validation_data=(val_data, val_labels),
                            epochs=epochs,
                            batch_size=batch_size,
                            verbose=1)

        # record scores
        scores[0].append(history.history['loss'])
        scores[1].append(history.history['acc'])
        scores[2].append(history.history['val_loss'])
        scores[3].append(history.history['val_acc'])

        print("")

    return scores

def main(args):
    average = []
    best = []

    # check for correct arguments
    if len(args) != 2:
        print("Usage: python DLNN.py data")
        exit()

    if K > 0:
        # process file
        with open(args[1], 'r') as f:
            train, test = kp.parse(f, shuffle=True, normalization=NORMALIZATION, split=SPLIT)
        
        # format training set
        train.data = kp.dataset.normalize(train.data, train.mean, train.std)
        train.labels = utils.to_categorical(train.labels)

        # format testing set
        test.data = kp.dataset.normalize(test.data, train.mean, train.std)
        test.labels = utils.to_categorical(test.labels)

        # perform K-fold cross-validation
        scores = k_fold_cross_validation(train.data, train.labels, K, EPOCHS, BATCH_SIZE)

        # average scores
        for score in scores:
            average.append(np.mean(score, axis=0))

        # find best scores
        best.append((np.argmin(average[0]) + 1, np.amin(average[0])))
        best.append((np.argmax(average[1]) + 1, np.amax(average[1])))
        best.append((np.argmin(average[2]) + 1, np.amin(average[2])))
        best.append((np.argmax(average[3]) + 1, np.amax(average[3])))

        # display averaged scores
        for i in range(EPOCHS):
            print("epoch {:d}".format(i + 1)
                  + " - loss: {:0.4f}".format(average[0][i])
                  + " - acc: {:0.4f}".format(average[1][i])
                  + " - val_loss: {:0.4f}".format(average[2][i])
                  + " - val_acc: {:0.4f}".format(average[3][i]))
        
        # display best scores
        print("\nbest"
              + " - loss: {:0.4f} (epoch {:d})".format(best[0][1], best[0][0])
              + " - acc: {:0.4f} (epoch {:d})".format(best[1][1], best[1][0])
              + " - val_loss: {:0.4f} (epoch {:d})".format(best[2][1], best[2][0])
              + " - val_acc: {:0.4f} (epoch {:d})".format(best[3][1], best[3][0]))

        # visualize training
        plot(np.arange(EPOCHS), average[0], average[1], average[2], average[3])
    else:
        # process file
        with open(args[1], 'r') as f:
            train, test = kp.parse(f, shuffle=True, normalization=NORMALIZATION, split=None)

        # format training set
        train.data = kp.dataset.normalize(train.data, train.mean, train.std)
        train.labels = utils.to_categorical(train.labels)

        # build model
        model = build_model(train.data.shape[1], train.labels.shape[1], summary=True)

        # train model
        model.fit(train.data, train.labels, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0)

        # save model
        model.save('models/DLNN.h5')

        # save data normalization parameters
        np.savez_compressed('models/DLNN', mean=train.mean, std=train.std)

if __name__ == '__main__':
    main(sys.argv)
