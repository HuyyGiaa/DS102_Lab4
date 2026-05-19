import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def evaluate_custom(y_true, y_pred):
    classes = np.unique(y_true)
    n_samples = len(y_true)
    
    accuracy = np.mean(y_true == y_pred)
    
    weighted_precision = 0.0
    weighted_recall = 0.0
    weighted_f1 = 0.0
    
    for c in classes:
        true_positives = np.sum((y_true == c) & (y_pred == c))
        false_positives = np.sum((y_true != c) & (y_pred == c))
        false_negatives = np.sum((y_true == c) & (y_pred != c))
        
        support = np.sum(y_true == c)
        weight = support / n_samples
        
        precision = true_positives / (true_positives + false_positives + 1e-9)
        recall = true_positives / (true_positives + false_negatives + 1e-9)
        
        if (precision + recall) > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
        else:
            f1 = 0.0
            
        weighted_precision += weight * precision
        weighted_recall += weight * recall
        weighted_f1 += weight * f1
        
    return accuracy, weighted_precision, weighted_recall, weighted_f1

def evaluate_sklearn(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    return accuracy, precision, recall, f1

