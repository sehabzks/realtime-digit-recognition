import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

model = load_model("mnist_dense.h5")
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

    # Görüntüyü işle
    roi_resized = cv2.resize(roi, (28, 28))
    roi_blurred = cv2.GaussianBlur(roi_resized, (5, 5), 0)
    roi_thresh = cv2.adaptiveThreshold(
        roi_blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )

    # Model için hazırla
    roi_normalized = roi_thresh.astype("float32") / 255.0
    roi_flat = roi_normalized.reshape(1, 784)

    # Tahmin
    predictions = model.predict(roi_flat, verbose=0)
    probabilities = tf.nn.softmax(predictions[0]).numpy()
    label = np.argmax(probabilities)
    confidence = np.max(probabilities)

    # Çizim ve gösterim
    cv2.rectangle(frame, (x1, y1), (x1 + box_size, y1 + box_size), (255, 0, 0), 2)
    cv2.putText(frame, f"Tahmin: {label} ({confidence*100:.1f}%)",
                (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # 3 pencere göster
    cv2.imshow("Canlı Kamera", frame)
    cv2.imshow("Model Girdisi", cv2.resize(roi_normalized, (200, 200)))
    cv2.imshow("Ters Threshold", cv2.resize(roi_thresh, (200, 200)))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
