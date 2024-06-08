from HandwritingDetection import HandwritingDetectionModel
from sklearn import datasets
from sklearn.model_selection import train_test_split
from Linear import Linear
from Sigmoid import Sigmoid
from ReLu import ReLu
from Softmax import Softmax
import matplotlib.pyplot as plt
import numpy as np

digits = datasets.load_digits()
data = digits.images.reshape((len(digits.images), -1))

X_train, X_test, y_train, y_test = train_test_split(data, digits.target, test_size=0.2, shuffle=False)

X_train = X_train.reshape(*X_train.shape, 1)
y_train = y_train.reshape(*y_train.shape, 1,  1)
X_test = X_test.reshape(*X_test.shape, 1)
y_test = y_test.reshape(*y_test.shape, 1, 1)

network = HandwritingDetectionModel([
    Linear(64, 32),
    Sigmoid(),
    Linear(32, 16),
    Sigmoid(),
    Linear(16, 10),
    Softmax()
])

network.train(X_train, y_train, 100, 0.01)
print(network.predict(X_test[0]))

preds = [np.argmax(network.predict(image)) for image in X_test]
preds = np.array(preds)

y_test = y_test.reshape(len(y_test))

correct = [pred == real for pred, real in zip(preds, y_test)]
_, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
for ax, image, prediction in zip(axes, X_test, preds):
    ax.set_axis_off()
    image = image.reshape(8, 8)
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
    ax.set_title(f"Prediction: {prediction}")
plt.show()
print(sum(correct)/len(y_test))



