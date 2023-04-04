# This is a sample Python script.
import cv2
import numpy as np



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

def BoundingBox(frame,corners,ids,draw = True):

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
        i_count = 0
        for i in range(len(corners)): #find centers of all detected arUco markers
            x_sum = corners_ordered[i][0][0][0] + corners_ordered[i][0][1][0] + corners_ordered[i][0][2][0] + corners_ordered[i][0][3][0]
            y_sum = corners_ordered[i][0][0][1] + corners_ordered[i][0][1][1] + corners_ordered[i][0][2][1] + corners_ordered[i][0][3][1]

            x_centerPixel = x_sum * .25
            y_centerPixel = y_sum * .25

            MarkerCenters[i] = (int(x_centerPixel),int(y_centerPixel))

            if i_count == 3: #draw box only when MarkerCenters list has been fully populated
                if draw == True:
                    cv2.line(frame, MarkerCenters[0], MarkerCenters[1], (0, 255, 0), thickness=2)
                    cv2.line(frame, MarkerCenters[1], MarkerCenters[2], (0, 255, 0), thickness=2)
                    cv2.line(frame, MarkerCenters[2], MarkerCenters[3], (0, 255, 0), thickness=2)
                    cv2.line(frame, MarkerCenters[3], MarkerCenters[0], (0, 255, 0), thickness=2)
            i_count = i_count + 1
        # print(MarkerCenters)
        return MarkerCenters

def PreprocessImage(frame, MarkerCenters):

    if not MarkerCenters:
        print('tuple is empty')


    else:
        # print('tuple is full')

        #arranging markers into clockwise order starting with top-left for input into cv2.GetPerspectiveTransform
        #topleft,topright,bottomright,bottomleft
        cornersArray = np.array(MarkerCenters)

        # import pdb; pdb.set_trace()

        # if cornersArray.shape[0] != None or cornersArray.shape[0] == 4: #HEEEELLLLLPPPP comparing to null is screwing me!!!!
        if cornersArray.shape[0] == 4:
            transInput = np.zeros([4, 2])
            sums = [0, 0, 0, 0]
            diffs = [0, 0, 0, 0]
            for i in range(len(MarkerCenters)):
                sums[i] = sum(MarkerCenters[i])
            for i in range(len(MarkerCenters)):
                diffs[i] = MarkerCenters[i][0] - MarkerCenters[i][1]

            # Top-left point will have the smallest sum.
            transInput[0, :] = MarkerCenters[sums.index(min(sums))]

            # Bottom-right point will have the largest sum.
            transInput[2, :] = MarkerCenters[sums.index(max(sums))]

            # Top-right point will have the smallest difference.
            transInput[1, :] = MarkerCenters[diffs.index(min(diffs))]

            # Bottom-left will have the largest difference.
            transInput[3, :] = MarkerCenters[diffs.index(max(diffs))]



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



            matrix = cv2.getPerspectiveTransform(np.float32(MarkerCenters), np.float32(destination_corners))
            input_image = cv2.warpPerspective(frame, matrix, (maxWidth, maxHeight))

            cv2.imshow('transformed feed', input_image)
            cv2.waitKey(1)

    return None

def main():

    cap = cv2.VideoCapture(0)

    iter_count = 0
    while True:
        success, frame = cap.read()

        corners, ids = findArucoMarkers(frame)
        MarkerCenters = BoundingBox(frame, corners=corners, ids=ids, draw=True)


        # import pdb;
        # pdb.set_trace()

        # print('markercenters = ',MarkerCenters)

        PreprocessImage(frame, MarkerCenters)

        cv2.imshow('Video Feed', frame)
        cv2.waitKey(1)
        iter_count = iter_count + 1


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
