import tensorflow as tf
from tensorflow.keras import layers, models, datasets
import matplotlib.pyplot as plt

# 1. MNIST veri setini yükle
(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()

# 2. Veriyi normalize et ve düzleştir (784 = 28x28)
x_train = x_train.reshape(-1, 784).astype("float32") / 255.0
x_test = x_test.reshape(-1, 784).astype("float32") / 255.0

# 3. Dense ağ modelini oluştur
model = models.Sequential([
    layers.Dense(256, activation='relu', input_shape=(784,)),
    layers.Dropout(0.3),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(10)  # 10 sınıf çıkışı
])

# 4. Modeli derle
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# 5. Modeli eğit
history = model.fit(x_train, y_train, epochs=5,
                    validation_data=(x_test, y_test))

# 6. Test doğruluğu
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f"Test doğruluğu: {test_acc:.4f}")

# 7. Modeli kaydet
model.save("mnist_dense.h5")
print("Model 'mnist_dense.h5' olarak kaydedildi.")

# 8. (Opsiyonel) Eğitim sürecini görselleştir
plt.plot(history.history['accuracy'], label='Eğitim Doğruluğu')
plt.plot(history.history['val_accuracy'], label='Doğrulama Doğruluğu')
plt.xlabel('Epoch')
plt.ylabel('Doğruluk')
plt.legend()
plt.show()
