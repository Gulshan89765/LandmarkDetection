import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D

 
data = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/facial_keypoints.csv')

 
data['Image'] = data['Image'].apply(lambda x: np.fromstring(x, sep=' '))

 data = data.dropna()

 X = np.vstack(data['Image'].values)
X = X.astype(np.float32) / 255.0  to [0, 1]
X = X.reshape(-1, 96, 96, 1)  


y = data[data.columns[:-1]].values  
y = (y - 48) / 48  


model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(96, 96, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.1))

model.add(Conv2D(64, (2, 2), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(500, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(90))  

 model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

 model.fit(X, y, epochs=50, validation_split=0.2)

 img = X[1].reshape(96, 96)  
pred = model.predict(X[1:2])[0]  
pred = pred * 48 + 48  

plt.imshow(img, cmap='gray')
plt.scatter(pred[0::2], pred[1::2], marker='x', color='red')  
plt.title("Predicted Facial Landmarks")
plt.show()
