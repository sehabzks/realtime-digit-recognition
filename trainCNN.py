import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

# 1. MNIST veri setini yükle
(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()

# 2. Veri setini ön işleme: yeniden boyutlandırma ve normalize etme
# CNN modeli 4 boyutlu giriş bekler: (num_samples, height, width, channels)
x_train = x_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0
x_test = x_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0

# 3. Basit CNN modelini oluştur
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10)  # 10 sınıf için çıkış
])

# 4. Modeli derle
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# 5. Modeli eğit
history = model.fit(x_train, y_train, epochs=5,
                    validation_data=(x_test, y_test))

# 6. Test verisi üzerinde modeli değerlendir
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f"Test doğruluğu: {test_acc:.4f}")

# 7. Eğitilmiş modeli kaydet
model.save("mnist_cnn.h5")
print("Model 'mnist_cnn.h5' olarak kaydedildi.")

# (Opsiyonel) Eğitim sürecini görselleştirme
plt.plot(history.history['accuracy'], label='Eğitim Doğruluğu')
plt.plot(history.history['val_accuracy'], label='Doğrulama Doğruluğu')
plt.xlabel('Epoch')
plt.ylabel('Doğruluk')
plt.legend(loc='lower right')
plt.show()
