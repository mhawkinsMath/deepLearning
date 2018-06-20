#!/home/math/tensorflow/venv/bin/python2.7

#Recurrent Neural Network example from watson course
# IMD sentiment data

from keras.preprocessing import sequence
from keras.models import Sequential 
from keras.layers  import Dense, Embedding
from keras.layers import LSTM
from keras.datasets import imdb

max_features =20000
maxlen =80

(x_train, y_train), (x_test,y_test) = imdb.load_data(num_words=max_features)

x_train = sequence.pad_sequences(x_train,maxlen=maxlen)
x_test= sequence.pad_sequences(x_test,maxlen=maxlen)

model = Sequential()
model.add(Embedding(max_features,128))
model.add(LSTM(128, dropout=0.2, recurrent_dropout =0.2))
model.add(Dense(1,activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=32, epochs=15, validation_data=(x_test,y_test))

model.evaluate(x_test, y_test, batch_size =32)



