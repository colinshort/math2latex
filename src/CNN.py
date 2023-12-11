import tensorflow as tf
import keras
from keras import Sequential
from keras.layers import Input, Dense, Dropout, Flatten, Conv2D, MaxPool2D
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import math
import pickle

df = pd.read_pickle('../data/data.pkl')
X = df.iloc[:,:-1].to_numpy()
Y = df.iloc[:,-1].to_numpy()
X = X.reshape(X.shape[0], int(math.sqrt(X.shape[1])), int(math.sqrt(X.shape[1])))

num_labels = max(Y) + 1

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

cnn_layers = [Input((45,45,1)),
Conv2D(256, 5, padding="same", activation="relu"),
MaxPool2D(),
Conv2D(128, 3, padding="same", activation="relu"),
MaxPool2D(),
Dropout(0.5),
Flatten(),
Dense(128, activation="relu"),
Dense(num_labels, activation="softmax")]
cnn_model = Sequential(cnn_layers)

print(cnn_model.summary())

cnn_model.compile(optimizer="adam", loss='sparse_categorical_crossentropy', metrics=['accuracy'])
n_epochs = 5
history = cnn_model.fit(x_train.reshape(-1, 45, 45 ,1), y_train, epochs=n_epochs,
                        validation_data=(x_test.reshape(-1, 45, 45 ,1), y_test))

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(np.arange(1, n_epochs+1), history.history['loss'], label='Train set')
plt.plot(np.arange(1, n_epochs+1), history.history['val_loss'], label='Test set')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(np.arange(1, n_epochs+1), history.history['accuracy'], label='Train set')
plt.plot(np.arange(1, n_epochs+1), history.history['val_accuracy'], label='Test set')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.grid(True)

plt.savefig("training_plots.jpg")

print(f"\nAccuracy on the final epoch of training was {100*history.history['accuracy'][-1]:0.2f}%")

cnn_scores = cnn_model.evaluate(x_test.reshape(-1, 45, 45 ,1), y_test)

print(f"\nThe CNN model achieves an accuracy of {cnn_scores[1]*100:.2f}% on the test data.")

pickle.dump(cnn_model, open('../models/cnn_model.pkl', 'wb'))