from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from data_preprocessing import combine_data, clean_data, train_test_split, normalize_data


def main():
    df = combine_data("winequality-red.csv", "winequality-white.csv")
    df = clean_data(df)
    
    x_train, y_train, x_test, y_test = train_test_split(df, test_size=0.2, random_state=42, target_col="quality")
    x_train, x_test = normalize_data(x_train, x_test)
    
    # Decision Tree
    dt_model = DecisionTreeClassifier(max_depth=10, min_samples_split=2)
    dt_model.fit(x_train, y_train)
    dt_y_pred = dt_model.predict(x_test)
    
    print("Decision Tree:")
    print("Accuracy:", accuracy_score(y_test, dt_y_pred))
    print("Precision:", precision_score(y_test, dt_y_pred, average='weighted', zero_division=0))
    print("Recall:", recall_score(y_test, dt_y_pred, average='weighted', zero_division=0))
    print("F1 Score:", f1_score(y_test, dt_y_pred, average='weighted', zero_division=0))
    
    # Random Forest
    rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_split=2)
    rf_model.fit(x_train, y_train)
    rf_y_pred = rf_model.predict(x_test)
    
    print("\nRandom Forest:")
    print("Accuracy:", accuracy_score(y_test, rf_y_pred))
    print("Precision:", precision_score(y_test, rf_y_pred, average='weighted', zero_division=0))
    print("Recall:", recall_score(y_test, rf_y_pred, average='weighted', zero_division=0))
    print("F1 Score:", f1_score(y_test, rf_y_pred, average='weighted', zero_division=0))
    
if __name__ == "__main__":
    main()