import h5py
import numpy              as np
import pandas             as pd
import tensorflow         as tf
import pickle             as pck
import sklearn_tda        as tda
import itertools

from skfeature                        import *
from metric_learn                     import *
from xgboost                          import *
from MKLpy.algorithms                 import *

from sklearn.preprocessing            import *
from sklearn.svm                      import *
from sklearn.ensemble                 import *
from sklearn.decomposition            import *
from sklearn.linear_model             import *
from sklearn.model_selection          import *
from sklearn.pipeline                 import *
from sklearn.manifold                 import *
from sklearn.neighbors                import *
from sklearn.metrics                  import *
from sklearn.metrics.pairwise         import *
from sklearn.feature_selection        import *
from sklearn.kernel_ridge             import *
from mpl_toolkits.mplot3d             import Axes3D

import matplotlib.pyplot  as plt
#%matplotlib notebook

def plot_regression_result(labels, pred):
    fig, axes = plt.subplots()
    axes.scatter(labels, pred, marker = "o", alpha = 1, c = "blue")
    axes.plot([min(labels),max(labels)],[min(labels),max(labels)])

def plot_confusion_matrix(cm, normalize = False, title = "Confusion matrix", cmap = plt.cm.Blues):
    if normalize:
        cm = cm.astype("float") / cm.sum(axis = 1)[:, np.newaxis]
    plt.figure()
    plt.imshow(cm, interpolation = "nearest", cmap = cmap)
    plt.title(title)
    plt.colorbar()
    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment = "center", 
                 color = "white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")

def diag_to_array(data):
    dataset, num_diag = [], len(data["0"].keys())
    for dim in data.keys():
        X = []
        for diag in range(num_diag):
            pers_diag = np.array(data[dim][str(diag)])
            X.append(pers_diag)
        dataset.append(X)
    return dataset

def diag_to_dict(D):
    X = dict()
    for f in D.keys():
        df = diag_to_array(D[f])
        for dim in range(len(df)):
            X[str(dim) + "_" + f] = df[dim]
    return X
