import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras import datasets

# 1. Kaydedilmiş modeli yükle
model = load_model("mnist_cnn.h5")
print("Model yüklendi.")

# 2. MNIST veri setini tekrar yükle (örnek görüntüler için)
(_, _), (x_test, y_test) = datasets.mnist.load_data()
x_test = x_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0

# 3. Test verisinden bir örnek seç ve tahmin yap
index = np.random.randint(0, len(x_test))  # Rastgele bir test verisi seç
 # Örneğin ilk görüntüyü kullanıyoruz, dilerseniz index değerini değiştirebilirsiniz.
sample_image = x_test[index]
sample_label = y_test[index]

# Modelin tahmin yapabilmesi için giriş boyutunu genişlet
sample_image_expanded = np.expand_dims(sample_image, axis=0)
predictions = model.predict(sample_image_expanded)
predicted_label = np.argmax(predictions[0])

print(f"Gerçek etiket: {sample_label}, Tahmin edilen etiket: {predicted_label}")

# 4. Örnek görüntüyü göster
plt.imshow(sample_image.squeeze(), cmap='gray')
plt.title(f"Gerçek: {sample_label} | Tahmin: {predicted_label}")
plt.axis('off')
plt.show()
