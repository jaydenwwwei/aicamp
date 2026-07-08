import cv2

capture=cv2.VideoCapture(0)  # Open the default camera (0)

while True:
    ret, frame = capture.read()
    cv2.imshow('My Camera', frame)
    key=cv2.waitKey(1000//240)
    if key>0:
        break
cv2.imshow('my image', cv2.imread(cv2.samples.findFile('My_camera_1.png')))
cv2.waitKey(0)