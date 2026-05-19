import numpy as np

import numpy as np

class Node:
    """Lưu trữ thông tin của một node trong cây quyết định"""
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, value=None):
        # Các thuộc tính cho node quyết định (decision node)
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        
        # Thuộc tính cho node lá (leaf node)
        self.value = value


class DecisionTree:
    def __init__(self, max_depth=10, min_samples_split=2, max_features=None):
        """
        Tham số max_features (int) sẽ được dùng khi xây dựng Random Forest. 
        Nếu để None, cây sẽ xét tất cả features tại mỗi lần chia (như Assignment 1).
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.root = None

    def fit(self, X, y):
        self.root = self._build_tree(X, y, depth=0)

    def _build_tree(self, X, y, depth):
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))

        # 1. Điều kiện dừng (Dừng khi đạt độ sâu tối đa, ít sample, hoặc các nhãn giống nhau hết)
        if (depth >= self.max_depth or n_samples < self.min_samples_split or n_classes == 1):
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        # 2. Xử lý max_features cho Random Forest
        feat_idxs = np.arange(n_features)
        if self.max_features is not None and self.max_features < n_features:
            np.random.shuffle(feat_idxs)
            feat_idxs = feat_idxs[:self.max_features]

        # 3. Tìm điểm chia (split) tốt nhất
        best_feat, best_thresh = self._best_split(X, y, feat_idxs)

        # Dừng nếu không tìm được cách chia nào tốt (ví dụ: các feature giống hệt nhau)
        if best_feat is None:
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        # 4. Tách dữ liệu ra 2 nhánh và gọi đệ quy để xây cây con
        left_idxs, right_idxs = self._split(X[:, best_feat], best_thresh)
        left_child = self._build_tree(X[left_idxs, :], y[left_idxs], depth + 1)
        right_child = self._build_tree(X[right_idxs, :], y[right_idxs], depth + 1)

        return Node(feature_index=best_feat, threshold=best_thresh, left=left_child, right=right_child)

    def _best_split(self, X, y, feat_idxs):
        best_gini = 1.0  # Gini cao nhất là 1, ta muốn tìm Gini thấp nhất
        best_feat, best_thresh = None, None

        # Chỉ duyệt qua các features được cho phép (hữu ích khi làm Random Forest)
        for feat_idx in feat_idxs:
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column) # Lấy các giá trị duy nhất làm ngưỡng chia

            for thr in thresholds:
                # Tính độ Gini Impurity cho cách chia này
                gini = self._gini_impurity_split(X_column, y, thr)

                if gini < best_gini:
                    best_gini = gini
                    best_feat = feat_idx
                    best_thresh = thr

        return best_feat, best_thresh

    def _gini_impurity_split(self, X_column, y, threshold):
        # Tách index ra 2 bên
        left_idxs, right_idxs = self._split(X_column, threshold)

        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 1.0 # Trả về mức Gini tệ nhất (1.0) nếu cách chia này không tách được dữ liệu

        # Tính toán độ hỗn loạn (Gini) của 2 nhánh
        n = len(y)
        n_l, n_r = len(left_idxs), len(right_idxs)
        gini_l, gini_r = self._gini(y[left_idxs]), self._gini(y[right_idxs])
        
        # Trọng số Gini của phép chia (Weighted Gini)
        child_gini = (n_l / n) * gini_l + (n_r / n) * gini_r
        return child_gini

    def _gini(self, y):
        # Công thức Gini tổng quát cho Multi-class: 1 - sum(p_i^2)
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        gini = 1.0 - np.sum(probabilities ** 2)
        return gini

    def _split(self, X_column, threshold):
        # numpy.argwhere trả về index thỏa mãn điều kiện
        left_idxs = np.argwhere(X_column <= threshold).flatten()
        right_idxs = np.argwhere(X_column > threshold).flatten()
        return left_idxs, right_idxs

    def _most_common_label(self, y):
        # Tìm Mode (nhãn xuất hiện nhiều nhất) cho nút lá (Multi-class/Binary đều dùng được)
        classes, counts = np.unique(y, return_counts=True)
        return classes[np.argmax(counts)]

    def predict(self, X):
        X = np.array(X)
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x, node):
        # Nếu đi đến nút lá, trả về giá trị dự đoán
        if node.value is not None:
            return node.value

        # Nếu giá trị feature nhỏ hơn ngưỡng, đi sang trái, ngược lại đi sang phải
        if x[node.feature_index] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)