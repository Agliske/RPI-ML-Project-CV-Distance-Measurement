import cv2
import cv2.aruco as aruco
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os



aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_50)
#
tag = np.zeros((300, 300, 1))
tag0 = cv2.aruco.drawMarker(aruco_dict, 0, 300, tag, 1)
tag1 = cv2.aruco.drawMarker(aruco_dict, 1, 300, tag, 1)
tag2 = cv2.aruco.drawMarker(aruco_dict, 2, 300, tag, 1)
tag3 = cv2.aruco.drawMarker(aruco_dict, 3, 300, tag, 1)

os.chdir(r"C:\Users\15039\Desktop\School Stuff\MachineLearning\Project\ProjectTest\Tags")
cv2.imshow('arUco marker', tag2)
cv2.imwrite('tag0.jpg', tag0)
cv2.imwrite('tag1.jpg', tag1)
cv2.imwrite('tag2.jpg', tag2)
cv2.imwrite('tag3.jpg', tag3)

# cv2.waitKey(0)

