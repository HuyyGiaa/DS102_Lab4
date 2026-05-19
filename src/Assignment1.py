from DecisionTree import DecisionTree
from data_preprocessing import combine_data, clean_data, train_test_split, normalize_data
from evaluate import accuracy_score, precision_score, recall_score, f1_score
def main():
    df = combine_data("winequality-red.csv", "winequality-white.csv")
    df = clean_data(df)
    
    x_train, y_train, x_test, y_test = train_test_split(df, test_size=0.2, random_state=42, target_col="quality")
    x_train, x_test = normalize_data(x_train, x_test)
    model = DecisionTree(max_depth=10, min_samples_split=2, max_features=None)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    print("Accuracy", accuracy_score(y_test, y_pred))
    print("Precision", precision_score(y_test, y_pred, average='weighted', zero_division=0))
    print("Recall", recall_score(y_test, y_pred, average='weighted', zero_division=0))
    print("F1 Score", f1_score(y_test, y_pred, average='weighted', zero_division=0))
    
if __name__ == "__main__":
    main()
