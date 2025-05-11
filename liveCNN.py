import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# 1. Eğitilmiş modeli yükle
model = load_model("mnist_cnn.h5")

# 2. Kamerayı başlat
cap = cv2.VideoCapture(0)

while True:
    # 3. Kameradan görüntüyü al
    ret, frame = cap.read()
    if not ret:
        break

    # 4. Griye çevir
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 5. Ortadan bir ROI (Region of Interest) seç (daha büyük bir alan)
    height, width = gray.shape
    center_x, center_y = width // 2, height // 2
    box_size = 200
    top_left_x = center_x - box_size // 2
    top_left_y = center_y - box_size // 2
    roi = gray[top_left_y:top_left_y + box_size, top_left_x:top_left_x + box_size]

    # 6. ROI'yi göster (kullanıcı nereye yazacağını bilsin)
    cv2.rectangle(frame, (top_left_x, top_left_y), (top_left_x + box_size, top_left_y + box_size), (255, 0, 0), 2)

    # 7. ROI’yi işlemeye hazırla
    roi_resized = cv2.resize(roi, (28, 28))
    roi_resized = cv2.bitwise_not(roi_resized)  # Beyaz rakam, siyah zemin varsayımı
    roi_resized = roi_resized.astype("float32") / 255.0
    roi_resized = np.expand_dims(roi_resized, axis=(0, -1))  # (1, 28, 28, 1)

    # 8. Tahmin yap ve softmax uygula
    predictions = model.predict(roi_resized, verbose=0)
    probabilities = tf.nn.softmax(predictions[0]).numpy()
    predicted_label = np.argmax(probabilities)
    confidence = np.max(probabilities)

    # 9. Sonucu ekrana yazdır
    label_text = f"Tahmin: {predicted_label} ({confidence*100:.1f}%)"
    cv2.putText(frame, label_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # 10. Girdiyi ve ana ekranı göster
    cv2.imshow("Model Girdisi", cv2.resize(roi_resized[0], (200, 200)))  # Küçük kutunun içi
    cv2.imshow("Canlı Rakam Tanıma", frame)

    # 11. Çıkış için 'q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# 12. Temizlik
cap.release()
cv2.destroyAllWindows()
