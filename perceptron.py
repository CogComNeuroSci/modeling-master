# initial code for fitting neural network models in scikit-learn environment

# import
from sklearn.neural_network import MLPClassifier
#from sklearn.linear_model import Perceptron
#from sklearn.linear_model import LogisticRegression
import numpy as np

# initialize
X = np.array([[0., 0.], [0., 1.], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 0])
clf1 = MLPClassifier(hidden_layer_sizes=(7),momentum=0,max_iter=20000)
#clf2 = LogisticRegression(C=100)
clf1.fit(X, y)
#clf2.fit(X, y)
print(clf1.coefs_,"\n",clf1.predict(X))
#print(clf2.intercept_, clf2.coef_)
#print(clf2.predict_proba(X), "\n", clf2.predict(X))