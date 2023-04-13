# This is a sample Python script.
import cv2
import numpy as np
import os
import tensorflow as tf


# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
def findArucoMarkers(frame, draw = True):

    greyFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    aruco_Dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
    aruco_params = cv2.aruco.DetectorParameters_create()
    corners, ids, rejected = cv2.aruco.detectMarkers(greyFrame, aruco_Dict, parameters=aruco_params)

    cameraMatrix = np.array([[1.18*10**3,0,7.2*10**2],[0,1.18*10**3,5.4*10**2],[0.0,0.0,1.0]])
    distMatrix = [-.2,.15,.006,-2.025*10**(-4),-.14]

    if draw == True:

        cv2.aruco.drawDetectedMarkers(frame, corners)

    return corners, ids

def boundingBox(frame,corners,ids,draw = True):

    cornersArray = np.array(corners)


    ids_ordered = [0,0,0,0]
    corners_ordered = [0,0,0,0]
    if cornersArray.shape[0] == 4: #if all 4 fiducials are detected, draw box

        i_count = 0
        for ArID in ids: #order the detected fiducials such that corners_ordered has the corners in id=0,1,2,3 order regardless of order they were detected

            corners_ordered[int(ArID)] = corners[i_count]
            ids_ordered[int(ArID)] = ids[i_count]
            i_count = i_count + 1

        # print('ids_ordered = ', ids_ordered)
        MarkerCenters = [0, 0, 0, 0]

        target1 = (corners_ordered[0][0][0][0], corners_ordered[0][0][0][1]) #
        target2 = (corners_ordered[1][0][1][0], corners_ordered[1][0][1][1])
        target3 = (corners_ordered[2][0][2][0], corners_ordered[2][0][2][1])
        target4 = (corners_ordered[3][0][3][0], corners_ordered[3][0][3][1])
        targetMarkers = [target1,target2,target3,target4]

        # MarkerCenters = [0, 0, 0, 0]
        # i_count = 0
        # for i in range(len(corners)): #find centers of all detected arUco markers
        #     x_sum = corners_ordered[i][0][0][0] + corners_ordered[i][0][1][0] + corners_ordered[i][0][2][0] + corners_ordered[i][0][3][0]
        #     y_sum = corners_ordered[i][0][0][1] + corners_ordered[i][0][1][1] + corners_ordered[i][0][2][1] + corners_ordered[i][0][3][1]
        #
        #     x_centerPixel = x_sum * .25
        #     y_centerPixel = y_sum * .25
        #
        #     MarkerCenters[i] = (int(x_centerPixel),int(y_centerPixel))
        #
        #     if i_count == 3: #draw box only when MarkerCenters list has been fully populated
        #         if draw == True:
        #             cv2.line(frame, MarkerCenters[0], MarkerCenters[1], (0, 255, 0), thickness=2)
        #             cv2.line(frame, MarkerCenters[1], MarkerCenters[2], (0, 255, 0), thickness=2)
        #             cv2.line(frame, MarkerCenters[2], MarkerCenters[3], (0, 255, 0), thickness=2)
        #             cv2.line(frame, MarkerCenters[3], MarkerCenters[0], (0, 255, 0), thickness=2)
        #     i_count = i_count + 1
        # print(MarkerCenters)
        if draw == True:
            cv2.line(frame, targetMarkers[0], targetMarkers[1], (0, 255, 0), thickness=2)
            cv2.line(frame, targetMarkers[1], targetMarkers[2], (0, 255, 0), thickness=2)
            cv2.line(frame, targetMarkers[2], targetMarkers[3], (0, 255, 0), thickness=2)
            cv2.line(frame, targetMarkers[3], targetMarkers[0], (0, 255, 0), thickness=2)
        return targetMarkers

def contourCleanup():
    return None

def preprocessImage(frame, targetMarkers):

    if not targetMarkers:
        print('tuple is empty')


    else:
        # print('tuple is full')

        #arranging markers into clockwise order starting with top-left for input into cv2.GetPerspectiveTransform
        #topleft,topright,bottomright,bottomleft
        cornersArray = np.array(targetMarkers)

        # import pdb; pdb.set_trace()

        # if cornersArray.shape[0] != None or cornersArray.shape[0] == 4: #HEEEELLLLLPPPP comparing to null is screwing me!!!!
        if cornersArray.shape[0] == 4:
            transInput = np.zeros([4, 2])
            sums = [0, 0, 0, 0]
            diffs = [0, 0, 0, 0]
            for i in range(len(targetMarkers)):
                sums[i] = sum(targetMarkers[i])
            for i in range(len(targetMarkers)):
                diffs[i] = targetMarkers[i][0] - targetMarkers[i][1]

            # Top-left point will have the smallest sum.
            transInput[0, :] = targetMarkers[sums.index(min(sums))]

            # Bottom-right point will have the largest sum.
            transInput[2, :] = targetMarkers[sums.index(max(sums))]

            # Top-right point will have the smallest difference.
            transInput[1, :] = targetMarkers[diffs.index(min(diffs))]

            # Bottom-left will have the largest difference.
            transInput[3, :] = targetMarkers[diffs.index(max(diffs))]



            # finding the maximum resolution for input image

            (tl, tr, br, bl) = transInput
            # Finding the maximum width.
            widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
            widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
            maxWidth = max(int(widthA), int(widthB))
            # Finding the maximum height.
            heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
            heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
            maxHeight = max(int(heightA), int(heightB))
            # Final destination co-ordinates.

            destination_corners = [[0, 0], [maxWidth, 0], [maxWidth, maxHeight], [0, maxHeight]]

            # print(destination_corners)



            matrix = cv2.getPerspectiveTransform(np.float32(targetMarkers), np.float32(destination_corners))
            snipped_image = cv2.warpPerspective(frame, matrix, (maxWidth, maxHeight))

            resized = cv2.resize(snipped_image,(int(9.3*30),int(10*30)))
            grey = cv2.cvtColor(resized,cv2.COLOR_BGR2GRAY)
            # blurred = cv2.GaussianBlur(grey,(5,5),0)

            avg_pixel_brightness = np.average(grey)
            junk, thresh = cv2.threshold(grey,avg_pixel_brightness,255,cv2.THRESH_BINARY_INV)

            # consider OTSU Threshold***

            # import pdb;
            # pdb.set_trace()

            contours, heirarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            # drawncont = cv2.drawContours(resized,contours,-1,(0,255,0), 1)

            cv2.imshow('thresh',thresh)

            # contourCleanup()

            rough_inputs = []
            iter_count = 0
            for cont in contours:

                current_img = thresh.copy()
                x,y,w,h = cv2.boundingRect(cont)
                drawnRec = cv2.rectangle(resized,(x,y),(x+w,y+h),(0,255,0),2)

                cropped_img = current_img[y:y+h, x:x+w]

                cont_area = cv2.contourArea(cont)
                # print(cont_area)
                if cont_area > 2500 and cont_area < 8000:
                    rough_inputs.append(cropped_img)

                iter_count = iter_count + 1

            # resizing image to 128x128
            inputs = []
            targetImageWidth = 128
            targetImageHeight = 128
            for image in rough_inputs:

                imWidth, imHeight = image.shape

                result = np.full((128, 128), 0, dtype=np.uint8)
                # result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

                x_center = (targetImageWidth - imWidth) // 2
                y_center = (targetImageHeight - imHeight) // 2

                result[x_center:x_center + imWidth, y_center:y_center + imHeight] = image

                inputs.append(result)

            # print(inputs)
            cv2.imshow('transformed feed', drawnRec)
            # cv2.imshow(inputs[0])

            # rows, cols = len(inputs), 1
            # for i in range(0, len(inputs), rows * cols):
            #     fig = plt.figure(figsize=(8, 8))
            #     for j in range(0, cols * rows):
            #         fig.add_subplot(rows, cols, j + 1)
            #         plt.imshow(inputs[i + j])
            #     plt.show()

            k = cv2.waitKey(1)
            return inputs





    return None

def main():
    os.chdir(r"C:\Users\15039\Desktop\School Stuff\MachineLearning\Project\ProjectTest\modelTraining")
    model = tf.keras.models.load_model('model1.h5')
    cap = cv2.VideoCapture(0)

    iter_count = 0
    while True:
        success, frame = cap.read()

        corners, ids = findArucoMarkers(frame,draw=False)
        targetMarkers = boundingBox(frame, corners=corners, ids=ids, draw=False)

        inputs = preprocessImage(frame, targetMarkers)


        if inputs:
            # print(inputs[0].shape) #the shape is clearly the correct shape at (128x128)
            # import pdb
            # pdb.set_trace()
            # outputs = model.predict(inputs[0])
            for image in inputs:

                outputs = model.predict(image.reshape((1,128,128))) #model.predict() needs an additional dimension of 1 on the (128,128) image
                print(outputs)

        cv2.imshow('Video Feed', frame)
        # cv2.imshow('inputs', inputs[0])


        k = cv2.waitKey(50)
        # if inputs:
        #     if k == ord('s'):  # wait for 's' key to save and exit
        #         cv2.imwrite('img' + str(iter_count) + '.png', inputs[0])
        #         print("image saved!")
        # iter_count = iter_count + 1



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/

