# packages
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

# set seed
seed = 314

def train_model(X, y, seed=seed):
    """
    Build a GBM on given data
    """
    model = GradientBoostingClassifier(
        learning_rate=0.02,
        n_estimators=550,
        max_depth=1.8,
        subsample=0.55,
        min_samples_leaf=45,
        random_state=seed
    )
    model.fit(X, y)
    return model