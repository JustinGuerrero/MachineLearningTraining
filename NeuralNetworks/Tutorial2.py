import tensorflow as tf 
from tensorflow import keras
import numpy as np 
from matplotlib import pyplot as plt 


''' this is a model taken from the Tensorflow website, I followed
along and recreated the model in Sublime text, and added a personal movie review
for the critically aclaimed "Cool Runnings"
'''

data = keras.datasets.imdb
#create data for the model
(train_data, train_labels), (test_data, test_labels) = data.load_data(num_words = 10000)

print(train_data[0])

#make markers for odd words, unknowns, end points, and unused
word_index = data.get_word_index() 
word_index = {k:(v+3) for k, v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2
word_index["UNUSED"] = 3

# we must reverse the words to work correctly with ML
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

#[process the data]
train_data = keras.preprocessing.sequence.pad_sequences(train_data, value=word_index["<PAD>"], padding="post", maxlen=250)
test_data = keras.preprocessing.sequence.pad_sequences(test_data, value=word_index["<PAD>"], padding="post", maxlen=250)


def decode_review(text):
	return " ".join([reverse_word_index.get(i, "?") for i in text])

'''#model 
# adding layers to the ML model
model = keras.Sequential()
model.add(keras.layers.Embedding(88000, 16))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation = "relu"))
model.add(keras.layers.Dense(1, activation="sigmoid"))

# give the summery, and the metrics, optimizer, and loss
model.summary()
model.compile(optimizer="adam", loss = "binary_crossentropy", metrics=["accuracy"])
# train data and labels
x_val = train_data[:10000]
x_train = train_data[10000:]

y_val = train_labels[:10000]
y_train = train_labels[10000:]
#create a fit model
fitmodel = model.fit(x_train, y_train, epochs=40, batch_size=512, validation_data=(x_val,y_val), verbose=1)

results = model.evaluate(test_data, test_labels)

print(results)
model.save("Model.h5")
'''


# now creating a model off of personal data found on IMDB for
# movie "cool runnings"

def review_encode(s):
	encoded = [1]

	for word in s:
		if word.lower() in word_index:
			encoded.append(word_index[word.lower()])
		else:
			encoded.append(2)

	return encoded

model = keras.models.load_model("Model.h5")

# we must take out the punctuations, and odd characters that will give off results
# by splitting and replacing them from the strings
with open("coolrunnings.txt", encoding="utf-8" ) as f:
	for line in f.readlines():
		nline = line.replace(",", "").replace(".", "").replace("(","").replace(")","").replace(":", "").replace("\"", "").strip().split(" ")
		encode = review_encode(nline)
		encode = keras.preprocessing.sequence.pad_sequences([encode], value = word_index["<PAD>"], padding="post", maxlen=250)
		predict = model.predict(encode)
		print(line)
		print(encode)
		print(predict[0])

# print results for model

#test_review = test_data[0]
#predict = model.predict([test_review])
#print(decode_review(test_review))
#print("prediction: " +str(predict[0]))
#print("actual: " + str(test_labels[0]))
#print(results)