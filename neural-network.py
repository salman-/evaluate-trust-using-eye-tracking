import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from Dataset import Dataset

dt = Dataset("Dataset.csv")
train_x, test_x, train_label, test_label = dt.get_train_test_data()

model = Sequential()
model.add(Input(shape=(train_x.shape[1],)))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(x=train_x, y=train_label, epochs=20, validation_data=(test_x, test_label))
