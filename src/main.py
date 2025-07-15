import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)
import matplotlib.pyplot as plt

# Modeller:
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC



def load_series_matrix(path):
    """
    GEO'dan indirilen series_matrix.txt dosyasını okur.
    - comment="!" ile başlayan tüm satırlar atlanır
    - index_col=0 ile ilk sütun (probe ID) satır indeksi olarak alınır
    """
    return pd.read_csv(path, sep="\t", comment="!", index_col=0)

def extract_labels_by_field(path, field_name):
    """
    !Sample_characteristics_ch1 satırlarında 'field_name:' içeren bölümleri bulur
    ve her örneğe karşılık gelen etiketleri döner.
    """
    char_lines = []
    with open(path, 'r') as f:
        for line in f:
            if line.startswith("!Sample_characteristics_ch1"):
                char_lines.append(line.strip())

    if not char_lines:
        raise ValueError("Hiç !Sample_characteristics_ch1 satırı bulunamadı.")

    target = None
    for line in char_lines:
        parts = line.split("\t")[1:]  # Tag'i atla
        texts = [p.strip().strip('"').lower() for p in parts]
        if any(p.startswith(field_name.lower() + ":") for p in texts):
            target = texts
            break

    if target is None:
        raise ValueError(f"'{field_name}' içeren bir satır bulunamadı.")

    # Her parçayı “label” kısmı olacak şekilde ':' işaretinden sonraya böl
    labels = [
        p.split(":", 1)[1].strip() if ":" in p else ""
        for p in target
    ]
    return pd.Series(labels, name=field_name.replace(" ", "_"))


if __name__ == "__main__":
    # 1) Veriyi oku ve etiketleri çıkar
    series_matrix_path = "GSE53757_series_matrix.txt"
    expr_df = load_series_matrix(series_matrix_path)
    y_tissue = extract_labels_by_field(series_matrix_path, "tissue")
    # “clear cell renal cell carcinoma” içerenleri tümör (1), diğerleri normal (0) olarak kodla
    y_binary = y_tissue.apply(lambda x: 1 if "carcinoma" in x.lower() else 0)
    X_full = expr_df.T  # (144 örnek × 54675 probe)
    print("Orijinal X shape (örnek × probe):", X_full.shape)

    # 2) Stratified K-Fold: 5 parçaya böl, dengeli sınıf dağılımı korunsun
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # 3) Denenecek farklı ‘en yüksek varyanslı N gen’ değerleri
    N_list = [5, 10, 20, 50, 100]

    # 4) Üç modelin hyperparametre ızgaraları (küçük ölçek, örnek amaçlı)
    param_grids = {
        "LR": {
            "logisticregression__C": [0.01, 1, 100],
            "logisticregression__penalty": ["l2"]
        },
        "RF": {
            "randomforestclassifier__n_estimators": [50, 100],
            "randomforestclassifier__max_depth": [None, 10]
        },
        "SVM": {
            "svc__C": [0.1, 1],
            "svc__gamma": ["scale", "auto"]
        }
    }

    # 5) Model tanımlayıcı fonksiyonlar (pipeline içinde kullanılacak)
    def make_lr():
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, random_state=42)
        )

    def make_rf():
        return make_pipeline(
            StandardScaler(),
            RandomForestClassifier(random_state=42, n_jobs=-1)
        )

    def make_svm():
        return make_pipeline(
            StandardScaler(),
            SVC(kernel="rbf", probability=True, random_state=42)
        )

    model_factories = {
        "LR": make_lr,
        "RF": make_rf,
        "SVM": make_svm
    }

    # 6) Sonuçları depolamak için sözlük: (N, model) → metrik listeleri
    cv_results = {
        (N, model_name): {
            "accuracy": [],
            "precision": [],
            "recall": [],
            "f1": [],
            "tn": [],
            "fp": [],
            "fn": [],
            "tp": []
        }
        for N in N_list
        for model_name in model_factories
    }

    # 7) Her N için, her fold’da:
    for N in N_list:
        print(f"\n--- En yüksek varyanslı {N} gen (5-fold CV) ---")
        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X_full, y_binary), start=1):
            # 7a) Eğitim/test böl
            X_train_fold = X_full.iloc[train_idx]
            y_train_fold = y_binary.iloc[train_idx]
            X_test_fold  = X_full.iloc[test_idx]
            y_test_fold  = y_binary.iloc[test_idx]

            # 7b) Eğitim seti üzerinde varyansa göre “Top N gen” seç
            variances = X_train_fold.var(axis=0)
            topN = variances.sort_values(ascending=False).head(N).index

            # 7c) Hem eğitim hem test setlerini bu N genle küçült
            X_tr_N = X_train_fold[topN]
            X_te_N = X_test_fold[topN]

            # 7d) Her model için: hyperparametre araması + test tahmini
            for model_name, factory in model_factories.items():
                print(f"  Fold {fold_idx}, Model {model_name}: Hiperparametre araması yapılıyor…")

                # Pipeline oluştur
                pipeline = factory()

                # GridSearchCV ile en iyi parametreleri bul, eğitim setinde 3-fold CV
                grid = GridSearchCV(
                    estimator=pipeline,
                    param_grid=param_grids[model_name],
                    cv=3,
                    scoring="accuracy",
                    n_jobs=-1,
                    verbose=0
                )
                grid.fit(X_tr_N, y_train_fold)
                best_model = grid.best_estimator_
                print(f"    En iyi parametreler: {grid.best_params_}")

                # 7e) Test seti üzerinde tahmin yap
                y_pred = best_model.predict(X_te_N)

                # 7f) Metrikleri hesapla
                acc = accuracy_score(y_test_fold, y_pred)
                prec = precision_score(y_test_fold, y_pred, zero_division=0)
                rec = recall_score(y_test_fold, y_pred, zero_division=0)
                f1 = f1_score(y_test_fold, y_pred, zero_division=0)
                tn, fp, fn, tp = confusion_matrix(y_test_fold, y_pred).ravel()

                # 7g) Sonuçları topla
                res_dict = cv_results[(N, model_name)]
                res_dict["accuracy"].append(acc)
                res_dict["precision"].append(prec)
                res_dict["recall"].append(rec)
                res_dict["f1"].append(f1)
                res_dict["tn"].append(tn)
                res_dict["fp"].append(fp)
                res_dict["fn"].append(fn)
                res_dict["tp"].append(tp)

                print(f"    Fold {fold_idx} sonuç → Acc: {acc:.3f}, Prec: {prec:.3f}, Rec: {rec:.3f}, F1: {f1:.3f}")
                print(f"    Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")

    # 8) Fold sonuçlarını özetle: ortalama ve standart sapma
    summary_rows = []
    for (N, model_name), metrics in cv_results.items():
        # “Ortalama confusion” için var olan toplam TN, FP, FN, TP’yi fold sayısına böl
        n_folds = skf.get_n_splits()
        avg_tn = np.mean(metrics["tn"])
        avg_fp = np.mean(metrics["fp"])
        avg_fn = np.mean(metrics["fn"])
        avg_tp = np.mean(metrics["tp"])

        summary_rows.append({
            "N": N,
            "Model": model_name,
            "Accuracy Mean": np.mean(metrics["accuracy"]),
            "Accuracy Std": np.std(metrics["accuracy"], ddof=1),
            "Precision Mean": np.mean(metrics["precision"]),
            "Precision Std": np.std(metrics["precision"], ddof=1),
            "Recall Mean": np.mean(metrics["recall"]),
            "Recall Std": np.std(metrics["recall"], ddof=1),
            "F1 Mean": np.mean(metrics["f1"]),
            "F1 Std": np.std(metrics["f1"], ddof=1),
            "Avg TN": avg_tn,
            "Avg FP": avg_fp,
            "Avg FN": avg_fn,
            "Avg TP": avg_tp
        })

    summary_df = pd.DataFrame(summary_rows)
    print("\n--- 5-Fold CV Sonuçları (Özet) ---")
    print(summary_df.sort_values(["N", "Model"]).reset_index(drop=True))

    # 9) Ortalamalara göre confusion matrix'leri kaydet
    for idx, row in summary_df.iterrows():
        N = int(row["N"])
        model_name = row["Model"]
        cm = np.array([[row["Avg TN"], row["Avg FP"]],
                       [row["Avg FN"], row["Avg TP"]]])

        fig, ax = plt.subplots(figsize=(4, 4))
        im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=np.max(summary_df[["Avg TN", "Avg FP", "Avg FN", "Avg TP"]].values))

        ax.set_title(f"CM (Avg) - N={N}, Model={model_name}")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Normal", "Tumor"])
        ax.set_yticklabels(["Normal", "Tumor"])

        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]:.1f}", ha="center", va="center", color="white", fontsize=12)

        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        plt.tight_layout()
        filename = f"confusion_matrix_N{N}_{model_name}.png"
        plt.savefig(filename)
        plt.close()
