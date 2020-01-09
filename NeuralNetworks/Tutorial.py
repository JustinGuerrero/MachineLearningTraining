import tensorflow as tf 
from tensorflow import keras
import numpy as np 
from matplotlib import pyplot as plt

''' practice model taken from Tim from Freecodecamp youtube video to 
start my training in Machine Learning. This is a basic example taken
from the Keras data inside of Tensorflows libraries.
'''

# import data sets from keras

data = keras.datasets.fashion_mnist

#split between testing and training

(train_images, train_labels), (test_images, test_labels) = data.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

train_images = train_images/255.0
test_images = test_images/255.0

# flatten the image 

model = keras.Sequential([
	keras.layers.Flatten(input_shape=(28,28)),
	keras.layers.Dense(128, activation="relu"),
	keras.layers.Dense(10, activation="softmax")
	])


model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])


#epoch is how many times a model sees the information
model.fit(train_images, train_labels, epochs=5)

# predict what the model is and use against what the model came out as
prediction = model.predict(test_images)

for i in range(5):
	plt.grid(False)
	plt.imshow(test_images[i], cmap=plt.cm.binary)
	plt.xlabel("Actual: " + class_names[test_labels[i]])
	plt.title("Prediction " + class_names[np.argmax(prediction[i])])
	plt.show()
#this will give us the index of the largest neuron
#print(class_names[np.argmax(prediction[0])])