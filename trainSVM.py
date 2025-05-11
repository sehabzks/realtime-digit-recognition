from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import joblib

# 1. MNIST verisini yükle
print("Veri yükleniyor...")
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X, y = mnist['data'], mnist['target'].astype(int)

# 2. Normalize et (0-255 arası yerine 0-1 arası)
X = X / 255.0

# 3. Eğitim ve test setlerine ayır
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. SVM modelini oluştur
print("Model eğitiliyor...")
clf = svm.SVC(kernel='rbf', gamma=0.05)  # RBF kernel daha iyi sonuç verir
clf.fit(X_train, y_train)

# 5. Test doğruluğu
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Test doğruluğu: {acc:.4f}")

# 6. Modeli kaydet
joblib.dump(clf, "svm_mnist_model.joblib")
print("Model 'svm_mnist_model.joblib' olarak kaydedildi.")
