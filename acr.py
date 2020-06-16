#!/usr/bin/env python3
###############################################################################

# Imports ---------------------------------------------------------------------
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier

# Constants --------------------------------------------------------------------
PRED_DATA = "sample_data_predict.txt"
TRAINING_DATA = "sample_data_training.txt"


# Functions  ------------------------------------------------------------------
def read_data(path):
    ret = pd.read_csv(path, sep="\t")
    ret = ret[sorted(ret.columns)]
    return ret


# Classes ---------------------------------------------------------------------
class AcrModel:
    """
    Model for predicting Acrs.
    """
    def __init__(self):
        self.__model = None

    def fit(self, X):
        """
        Fit a random forest model.
        """
        weights = X.weight.tolist()
        y = X.y.tolist()
        X = X.drop(columns=["y", "weight", "name"])

        self.__model = ExtraTreesClassifier(
            n_estimators=1000,
            random_state=123890,
        )

        self.__model.fit(
            X, y,
            sample_weight=weights,
        )

    def score(self, X):
        """
        Returns scores for input Acr candidates.
        """
        names = X.name.tolist()
        X = X.drop(columns="name")
        scores = {
            x: y for x, y in
            zip(names, self.__model.predict_proba(X)[:, 1])
        }

        names = sorted(names, key=lambda x: int(x.split("_")[-1]), reverse=True)
        return zip(
            names,
            [scores[x] for x in names]
        )


# Main ------------------------------------------------------------------------
# Train the model
X = read_data(TRAINING_DATA)
model = AcrModel()
model.fit(X)

# Test the model
X = read_data(PRED_DATA)
print("\n".join(["{}\t{}".format(x, y) for x, y in model.score(X)]))
