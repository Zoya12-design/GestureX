import warnings
from pathlib import Path

import joblib
import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report, f1_score,
                             precision_score, recall_score)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "landmarks_data"
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

# ---- Data load: sab logon ki files jodo ----
files = sorted(
    f for f in DATA_DIR.glob("landmarks_*.csv")
    if not f.stem.endswith("_backup")
)
if not files:
    raise SystemExit("landmarks_data mein koi landmarks_<naam>.csv nahi mili.")
print("Files:", [f.name for f in files])
df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
features = [f"f{i}" for i in range(63)]

print("\nGesture counts:\n", df["label"].value_counts())
print("\nPerson counts:\n", df["person"].value_counts(), "\n")

# ---- Train / test split ----
persons = sorted(df["person"].unique())
if len(persons) >= 2:
    # Aakhri insaan sirf test ke liye (model ne kabhi nahi dekha)
    test_person = persons[-1]
    train = df[df["person"] != test_person]
    test = df[df["person"] == test_person]
    print(f"Test person: {test_person} | Train persons: {persons[:-1]}")
else:
    # Ek hi insaan: har gesture ke pehle 80% frames train, aakhri 20% test
    tr, te = [], []
    for _, g in df.groupby("label", sort=False):
        cut = int(len(g) * 0.8)
        tr.append(g.iloc[:cut])
        te.append(g.iloc[cut:])
    train, test = pd.concat(tr), pd.concat(te)
    print("Sirf ek insaan ka data hai, time-ordered split use hua "
          "(natija asli generalization nahi dikhata).")

X_train, y_train = train[features], train["label"]
X_test, y_test = test[features], test["label"]

# ---- Models ----
models = {
    "KNN": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
    "SVM": make_pipeline(StandardScaler(),
                         SVC(kernel="rbf", C=10, probability=True, random_state=42)),
    "MLP": make_pipeline(StandardScaler(),
                         MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500,
                                       random_state=42)),
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

best = results.sort_values("F1-score (%)", ascending=False, kind="stable").iloc[0]["Model"]
print("\nBest model:", best)
print(classification_report(y_test, preds[best], zero_division=0))

# ---- Graphs ----
fig, ax = plt.subplots(figsize=(7, 6))
ConfusionMatrixDisplay.from_predictions(y_test, preds[best], ax=ax,
                                        xticks_rotation=45, cmap="Blues")
ax.set_title(f"Confusion Matrix - {best}")
plt.tight_layout()
plt.savefig(RESULTS / "confusion_matrix.png", dpi=150)
plt.close()

ax = results.set_index("Model")[["Accuracy (%)", "F1-score (%)"]].plot(
    kind="bar", figsize=(7, 5))
ax.set_ylim(0, 105)
ax.set_title("Model Comparison")
ax.set_ylabel("%")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(RESULTS / "model_comparison.png", dpi=150)
plt.close()
print("Graphs saved in:", RESULTS)

# ---- Best model ko poore data par dobara train karke save karo ----
final = models[best]
final.fit(df[features], df["label"])
out = ROOT / "models"
out.mkdir(exist_ok=True)
joblib.dump(final, out / "gesture_landmark_model.pkl")
print("Saved model:", out / "gesture_landmark_model.pkl")