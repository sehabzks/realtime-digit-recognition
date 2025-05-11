import cv2
import numpy as np
import joblib

# 1. Eğitilmiş modeli yükle
model = joblib.load("svm_mnist_model.joblib")

# 2. Kamerayı başlat
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    h, w = gray.shape
    cx, cy = w // 2, h // 2
    box_size = 200
    x1, y1 = cx - box_size // 2, cy - box_size // 2
    roi = gray[y1:y1 + box_size, x1:x1 + box_size]

    # ROI kutusunu göster
    cv2.rectangle(frame, (x1, y1), (x1 + box_size, y1 + box_size), (255, 0, 0), 2)

    # Görüntü işleme
    roi_resized = cv2.resize(roi, (28, 28))
    roi_inverted = cv2.bitwise_not(roi_resized)
    roi_blurred = cv2.GaussianBlur(roi_inverted, (5, 5), 0)

    # Normalize ve düzleştir
    roi_norm = roi_blurred.astype("float32") / 255.0
    roi_flat = roi_norm.reshape(1, -1)

    # Tahmin
    prediction = model.predict(roi_flat)[0]

    # Sonucu ekrana yaz
    cv2.putText(frame, f"Tahmin: {prediction}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Görselleri göster
    cv2.imshow("Canlı Kamera (SVM)", frame)
    cv2.imshow("Model Girdisi", cv2.resize(roi_norm, (200, 200)))
    cv2.imshow("Ters Görüntü", cv2.resize(roi_inverted, (200, 200)))

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
