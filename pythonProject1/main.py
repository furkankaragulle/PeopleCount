import cv2


cap = cv2.VideoCapture(0)  # 0 varsayılan kamera indeksi

# OpenCV vücut tespit modeli
body_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_fullbody.xml')

# İnsan sayısı değişkeni
insan_sayisi = 0

while True:
    # Kamera görüntüsünü al
    ret, frame = cap.read()

    # Görüntü okunamazsa döngüyü sonlandır
    if not ret:
        break

    # Gri tona dönüştürmek adına
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Vücutları tespit et
    bodies = body_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(30, 30))

    # Her yeni vücut için sayıyı artır
    for (x, y, w, h) in bodies:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        insan_sayisi += 1

    # İnsan sayısını ekrana yazdır
    cv2.putText(frame, f'Insan Sayisi: {insan_sayisi}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    # Kameranın görüntüsünü göster
    cv2.imshow('Human Detection', frame)

    # Çıkış için 'x' tuşuna veya başka bir tuşa basın
    key = cv2.waitKey(1)
    if key == ord('x') or key != -1:
        break

# Video akışını serbest bırak ve pencereyi kapat
cap.release()
cv2.destroyAllWindows()
