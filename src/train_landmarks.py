from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             f1_score, precision_score, recall_score)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parent.parent
df = pd.read_csv(ROOT / "landmarks_data" / "landmarks.csv")
features = [f"f{i}" for i in range(63)]

print(df["label"].value_counts(), "\n")
print(df["person"].value_counts(), "\n")

persons = sorted(df["person"].unique())
if len(persons) >= 2:
    # Aakhri insaan ka data sirf test ke liye (asli evaluation)
    test_person = persons[-1]
    train = df[df["person"] != test_person]
    test = df[df["person"] == test_person]
    print("Test person:", test_person)
else:
    # Ek hi insaan: har gesture ke pehle 80% frames train, aakhri 20% test
    # (shuffle nahi, warna ek jaise frames dono mein aa jate hain)
    tr, te = [], []
    for _, g in df.groupby("label", sort=False):
        cut = int(len(g) * 0.8)
        tr.append(g.iloc[:cut])
        te.append(g.iloc[cut:])
    train, test = pd.concat(tr), pd.concat(te)
    print("Sirf ek insaan ka data hai, time-ordered split use hua.")

X_train, y_train = train[features], train["label"]
X_test, y_test = test[features], test["label"]

models = {
    "KNN": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
    "SVM": make_pipeline(StandardScaler(), SVC(kernel="rbf", C=10, probability=True, random_state=42)),
}

rows, preds = [], {}
for name, model in models.items():
    model.fit(X_train, y_train)
    p = model.predict(X_test)
    preds[name] = p
    rows.append({
        "Model": name,
        "Accuracy (%)": round(accuracy_score(y_test, p) * 100, 2),
        "Precision (%)": round(precision_score(y_test, p, average="macro", zero_division=0) * 100, 2),
        "Recall (%)": round(recall_score(y_test, p, average="macro", zero_division=0) * 100, 2),
        "F1-score (%)": round(f1_score(y_test, p, average="macro", zero_division=0) * 100, 2),
    })

results = pd.DataFrame(rows)
print("\n", results.to_string(index=False))
results.to_csv(ROOT / "model_comparison_landmarks.csv", index=False)

best = results.sort_values("F1-score (%)", ascending=False).iloc[0]["Model"]
print("\nBest model:", best)
print(classification_report(y_test, preds[best], zero_division=0))

# Best model ko poore data par dobara train karke save karo
final = models[best]
final.fit(df[features], df["label"])
out = ROOT / "models"
out.mkdir(exist_ok=True)
joblib.dump(final, out / "gesture_landmark_model.pkl")
print("Saved:", out / "gesture_landmark_model.pkl")