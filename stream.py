import cv2
import requests

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    _, img = cv2.imencode('.jpg', frame)
    requests.post("http://localhost:8000/upload", files={"file": ("frame.jpg", img.tobytes(), "image/jpeg")})
