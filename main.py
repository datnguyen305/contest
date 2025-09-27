from src.models import (
    SVM,
    NaiveBayes,
    KNN, KNNRegressor,
    LR,
    RandomForestClassifier, RFRegressor,
    AdaBoostKNN,
    AdaBoostLR,
    AdaBoostRandomForest,
    AdaBoostSVM,
    AdaBoostNB,
    BaggingSVM,
    BaggingNB,
    BaggingKNN,
    BaggingLR,
    BaggingRF,
    RandomSubspaceKNN,
    RandomSubspaceLR,
    RandomSubspaceRF,
    RandomSubspaceSVM,
    RandomSubspaceNB,
    StackingEnsemble
)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import f1_score

if __name__ == "__main__":
    # Đọc dữ liệu
    df = pd.read_csv("train.csv")
    # Giả sử cột đặc trưng là từ 0 đến 38, cột label là 'label_num' hoặc 'Label'
    if 'label_num' in df.columns:
        y = df['label_num']
    elif 'Label' in df.columns:
        y = df['Label']
    else:
        raise ValueError("Không tìm thấy cột label phù hợp!")
    X = df.iloc[:, :39]

    # Chia tập train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    models = [
        ("KNN", KNN()),
        ("LR", LR()),
        ("NB", NaiveBayes()),
        ("RF", RandomForestClassifier()),
        ("SVM", SVM()),
        ("Adaboost_KNN", AdaBoostKNN()),
        ("Adaboost_LR", AdaBoostLR()),
        ("Adaboost_NB", AdaBoostNB()),
        ("Adaboost_RF", AdaBoostRandomForest()),
        ("Adaboost_SVM", AdaBoostSVM()),
        ("Bagging_KNN", BaggingKNN()),
        ("Bagging_NB", BaggingNB()),
        ("Bagging_LR", BaggingLR()),
        ("Bagging_RF", BaggingRF()),
        ("Bagging_SVM", BaggingSVM()),
        ("RS_KNN", RandomSubspaceKNN()),
        ("RS_LR", RandomSubspaceLR()),
        ("RS_NB", RandomSubspaceNB()),
        ("RS_RF", RandomSubspaceRF()),
        ("RS_SVM", RandomSubspaceSVM()),
    ]

    results = []
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for name, model in models:
        print(f"\nTraining {name} with 5-fold CV...")
        f1_scores = []
        for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
            model.fit(X_tr, y_tr)
            y_pred = model.predict(X_val)
            f1 = f1_score(y_val, y_pred, average='macro')
            f1_scores.append(f1)
            print(f"  Fold {fold+1}: F1-score = {f1:.4f}")
        mean_f1 = np.mean(f1_scores)
        results.append((name, mean_f1))
        print(f"{name} Mean F1-score (5-fold): {mean_f1:.4f}")

    # Lưu kết quả ra file CSV
    results_df = pd.DataFrame(results, columns=["Model", "Mean_F1"])
    results_df.to_csv("model_f1_results.csv", index=False)
    print("\nKết quả đã được lưu vào model_f1_results.csv")

    #Lưu trọng số mô hình tốt nhất, vào file best_model.pkl
    best_model_name = results_df.loc[results_df["Mean_F1"].idxmax(), "Model"]
    best_model = dict(models)[best_model_name]
    import joblib
    joblib.dump(best_model, "best_model.pkl")
    

    # Trực quan hóa kết quả
    plt.figure(figsize=(12,6))
    plt.bar(results_df["Model"], results_df["Mean_F1"], color='skyblue')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel("Mean F1-score (5-fold)")
    plt.title("So sánh F1-score các mô hình")
    plt.tight_layout()
    plt.show()

    # Chọn model tốt nhất
    best_model_name = results_df.loc[results_df["Mean_F1"].idxmax(), "Model"]
    best_f1 = results_df["Mean_F1"].max()
    print(f"\nModel tốt nhất: {best_model_name} với F1-score = {best_f1:.4f}")
