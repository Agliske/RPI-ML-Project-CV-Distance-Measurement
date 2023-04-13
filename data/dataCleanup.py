import numpy as np
import cv2
import os

compile_images = False
insert_keys = False
combine_all_data = True

if compile_images == True:

    triangles = []

    folder_dir = r"C:\Users\15039\Desktop\School Stuff\MachineLearning\Project\ProjectTest\data\quadrilaterals"
    targetImageHeight = 128
    targetImageWidth = 128
    for photo in os.listdir(folder_dir):
        image = cv2.imread(r'C:/Users/15039/Desktop/School Stuff/MachineLearning/Project/ProjectTest/data/quadrilaterals/'+ str(photo))

        imWidth,imHeight, channels = image.shape

        result = np.full((128,128,3),(0,0,0),dtype=np.uint8)
        x_center = (targetImageWidth - imWidth) // 2
        y_center = (targetImageHeight - imHeight) // 2

        result[x_center:x_center + imWidth, y_center:y_center + imHeight] = image
        result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        cv2.imshow('result',result)
        cv2.waitKey(50)

        triangles.append(result)

    triangles = np.array(triangles)
    print(triangles.shape)
    os.chdir(r"C:\Users\15039\Desktop\School Stuff\MachineLearning\Project\ProjectTest\data")
    np.save('quadrilaterals',triangles,True)

if insert_keys == True:
    os.chdir(r"C:\Users\15039\Desktop\School Stuff\MachineLearning\Project\ProjectTest\data")
    shape_pics = np.load(r"C:\Users\15039\Desktop\School Stuff\MachineLearning\Project\ProjectTest\data\1_imageArrays\triangles.npy",allow_pickle=True)

    keys = np.zeros((1,shape_pics.shape[0])) #circles are zeros, hexagons are ones, quads are twos, triangles are threes.
    keys.fill(3)
    # print(keys)

    Data = (keys, shape_pics)

    Data =np.array(Data,dtype=object)
    np.save('triangleData', Data, True)

if combine_all_data == True:
    os.chdir(r"C:\Users\15039\Desktop\School Stuff\MachineLearning\Project\ProjectTest\data")
    circ_dataset = np.load(r"C:\Users\15039\Desktop\School Stuff\MachineLearning\Project\ProjectTest\data\circleData.npy", allow_pickle=True)
    circ_keys = circ_dataset[0]
    circles = circ_dataset[1]

    hex_dataset = np.load(r"C:\Users\15039\Desktop\School Stuff\MachineLearning\Project\ProjectTest\data\hexagonData.npy", allow_pickle=True)
    hex_keys = hex_dataset[0]
    hexagons = hex_dataset[1]

    quad_dataset = np.load(r"C:\Users\15039\Desktop\School Stuff\MachineLearning\Project\ProjectTest\data\quadrilateralData.npy", allow_pickle=True)
    quad_keys = quad_dataset[0]
    quads = quad_dataset[1]

    tri_dataset = np.load(r"C:\Users\15039\Desktop\School Stuff\MachineLearning\Project\ProjectTest\data\triangleData.npy", allow_pickle=True)
    tri_keys = tri_dataset[0]
    triangles = tri_dataset[1]

    total_keys = np.concatenate((circ_keys,hex_keys,quad_keys,tri_keys), axis=1)
    total_images = np.concatenate((circles,hexagons,quads,triangles))
    # print(total_keys)
    # print(total_images.shape)

    total_data = (total_keys,total_images)
    total_data = np.array(total_data,dtype=object)
    np.save('totalData',total_data,allow_pickle=True)














