import time
import platform
import os
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# --- Dataset ---
X, y = make_classification(
    n_samples=50000,
    n_features=64,
    n_informative=40,
    n_redundant=10,
    n_classes=10,
    n_clusters_per_class=1,
    random_state=42,
)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Model ---
clf = RandomForestClassifier(n_estimators=500,
n_jobs=-1, random_state=42)

# --- Benchmark ---
print(f"Machine : {platform.node()}")
print(f"CPUs    : {os.cpu_count()}")
print(f"Samples : {len(X_train):,} train / {len(X_test):,} test")
print(f"Model   : Random Forest, 500 trees\n")

t0 = time.perf_counter()
clf.fit(X_train, y_train)
train_time = time.perf_counter() - t0

t1 = time.perf_counter()
preds = clf.predict(X_test)
pred_time = time.perf_counter() - t1

acc = accuracy_score(y_test, preds)

print(f"Training time : {train_time:.3f} s")
print(f"Predict time  : {pred_time:.4f} s")
print(f"Accuracy      : {acc*100:.2f} %")