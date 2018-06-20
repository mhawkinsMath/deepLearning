#!/home/math/tensorflow/venv/bin/python2.7

#beyond sequential models -- non-sequential models and the functional API watson course
#similiar to sequential model from first keras program

from keras.layers import Input,Dense
from keras.models import Model

num_classes = 10 
inputs = Input(shape=(784,))

x=Dense(512, activation='relu')(inputs)
x=Dense(512, activation='relu')(x)
predications=Dense(num_classes, activation ='softmax')(x)

model= Model(inputs=inputs, outputs=predications)
model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(....) #same as beffore??




