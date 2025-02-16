import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical

iris = load_iris()
X = iris.data
y = iris.target

X = X.reshape((X.shape[0], 1, 1, 4))

y = to_categorical(y, num_classes=3)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train = X_train / 255.0
X_test = X_test / 255.0

model_cnn = Sequential([
    Conv2D(32, (1, 1), activation='relu', input_shape=(1, 1, 4)),  # Image 1x1 avec 4 canaux
    MaxPooling2D(pool_size=(1, 1)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(3, activation='softmax')  # 3 classes : setosa, versicolor, virginica
])

model_cnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model_cnn.fit(X_train, y_train, epochs=50, batch_size=16, validation_split=0.2)

model_cnn.save("cnn_model.h5")
