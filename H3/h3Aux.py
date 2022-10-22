#wo MLPs – 𝑀𝐿𝑃1 and 𝑀𝐿𝑃2 – each with two hidden layers of size 10, hyperbolic tangent
#function as the activation function of all nodes, a maximum of 500 iterations, and a fixed
#seed (random_state=0). 𝑀𝐿𝑃1 should be parameterized with early stopping while 𝑀𝐿𝑃2
#should not consider early stopping. Remaining parameters (e.g., loss function, batch size,
#regularization term, solver) should be set as default.

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import log_loss
from sklearn.metrics import brier_score_loss
from sklearn.metrics import hinge_loss
from sklearn.metrics import zero_one_loss
from sklearn.metrics import jaccard_score
from sklearn.metrics import hamming_loss
from sklearn.metrics import zero_one_loss
from sklearn.metrics import fbeta_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import multilabel_confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import time
import sys
from sklearn.linear_model import LinearRegression, Ridge, Lasso
import pandas as pd
from scipy.io.arff import loadarff
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics, datasets

# Load data

data = loadarff('kin8nm.arff')
df = pd.DataFrame(data[0])
df.head()

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 0)

# MLP1

mlp1 = MLPClassifier(hidden_layer_sizes=(10,10), activation='tanh', solver='adam', alpha=0.0001, batch_size='auto', learning_rate='constant', learning_rate_init=0.001, power_t=0.5, max_iter=500, shuffle=True, random_state=0, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True, early_stopping=True, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08, n_iter_no_change=10)

# MLP2

mlp2 = MLPClassifier(hidden_layer_sizes=(10,10), activation='tanh', solver='adam', alpha=0.0001, batch_size='auto', learning_rate='constant', learning_rate_init=0.001, power_t=0.5, max_iter=500, shuffle=True, random_state=0, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08, n_iter_no_change=10)

# Fit the model

mlp1.fit(X_train, y_train)
mlp2.fit(X_train, y_train)

# Predict the model

y_pred_mlp1 = mlp1.predict(X_test)
y_pred_mlp2 = mlp2.predict(X_test)

# Accuracy

print("Accuracy MLP1: ", accuracy_score(y_test, y_pred_mlp1))
print("Accuracy MLP2: ", accuracy_score(y_test, y_pred_mlp2))

# Confusion Matrix

print("Confusion Matrix MLP1: ", confusion_matrix(y_test, y_pred_mlp1))
print("Confusion Matrix MLP2: ", confusion_matrix(y_test, y_pred_mlp2))
