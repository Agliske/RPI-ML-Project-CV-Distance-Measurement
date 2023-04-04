import cv2
import os

cap = cv2.VideoCapture(0)

num = 0
os.chdir(r"C:\Users\15039\Desktop\School Stuff\MachineLearning\Project\ProjectTest\CameraCalibration\images")

while cap.isOpened():

    succes, img = cap.read()

    k = cv2.waitKey(5)

    if k == 27:
        break
    elif k == ord('s'): # wait for 's' key to save and exit
        os.chdir(r"C:\Users\15039\Desktop\School Stuff\MachineLearning\Project\ProjectTest\CameraCalibration\images")
        cv2.imwrite('images/img' + str(num) + '.png', img)
        print("image saved!")
        num += 1

    cv2.imshow('Img',img)

# Release and destroy all windows before termination
cap.release()

cv2.destroyAllWindows()