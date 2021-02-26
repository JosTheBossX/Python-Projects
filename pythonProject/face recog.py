import cv2
import numpy as np
import face_recognition

imgJos = face_recognition.load_image_file('IMAGE/jos2.jpg')
imgJos = cv2.cvtColor(imgJos, cv2.COLOR_BGR2RGB)
imgTest = face_recognition.load_image_file('IMAGE/jos1.jpg')
imgTest = cv2.cvtColor(imgTest, cv2.COLOR_BGR2RGB)

faceLoc = face_recognition.face_locations(imgJos)[0]
encodeJos = face_recognition.face_encodings(imgJos)[0]
cv2.rectangle(imgJos, (faceLoc[3], faceLoc[0]), (faceLoc[1], faceLoc[2]), (255, 0, 255), 2)

faceLocTest = face_recognition.face_locations(imgTest)[0]
encodeTest = face_recognition.face_encodings(imgTest)[0]
cv2.rectangle(imgTest, (faceLocTest[3], faceLocTest[0]), (faceLocTest[1], faceLocTest[2]), (255, 0, 255), 2)

results = face_recognition.compare_faces([encodeJos], encodeTest)
faceDis = face_recognition.face_distance([encodeJos], encodeTest)
print(results, faceDis)
cv2.putText(imgTest, f'{results}{round(faceDis[0],2)}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

cv2.imshow('Jos', imgJos)
cv2.imshow('Jos Test', imgTest)
cv2.waitKey(0)