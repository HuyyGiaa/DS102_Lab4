from DecisionTree import DecisionTree
import numpy as np

class RandomForest:
    def __init__(self, n_estimators=10, max_depth=10, min_samples_split=2, max_features="sqrt"):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.trees = []

    def fit(self, X, y):
        self.trees = []
        n_samples, n_features = X.shape

        if self.max_features == "sqrt":
            max_feat_int = int(np.sqrt(n_features))
        elif isinstance(self.max_features, int):
            max_feat_int = self.max_features
        else:
            max_feat_int = n_features

        for _ in range(self.n_estimators):
            idxs = np.random.choice(n_samples, n_samples, replace=True)
            X_bootstrap = X[idxs]
            y_bootstrap = y[idxs]

            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                max_features=max_feat_int 
            )
            
            tree.fit(X_bootstrap, y_bootstrap)
            self.trees.append(tree)

    def predict(self, X):
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        tree_preds = np.swapaxes(tree_preds, 0, 1)

        y_pred = [self._most_common_label(preds) for preds in tree_preds]
        return np.array(y_pred)

    def _most_common_label(self, y):
        classes, counts = np.unique(y, return_counts=True)
        return classes[np.argmax(counts)]