import cv2 as cv
import open3d as o3d
import numpy as np
from mayavi import mlab
import os as os


def initialise():
    return 0


def importFrames():
    importFolder = r"C:\Users\84307\Desktop\UNI\Year 3\ELEC0036\Project\Project I\importImages"
    exportFolder = r"C:\Users\84307\Desktop\UNI\Year 3\ELEC0036\Project\Project I\grayImages"
    files = [file for file in os.listdir(importFolder) if os.path.isfile(os.path.join(importFolder, file))]
    images = []
    grayImages = []

    for file in files:
        image = cv.imread(os.path.join(importFolder, file))
        images.append(image)
        grayImage = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        grayImages.append(grayImage)
        path = os.path.join(exportFolder, file)
        cv.imwrite(path, grayImage)


def getCameraParams():
    W = 4032
    H = 3024
    F = 26
    K = np.array([[3385,0,2025],[0,3385,1499],[0,0,1]])
    return W, H, F, K


def ORBDetect():
    #sift = cv.SIFT_create(0, 3, 0.04, 10)
    orb = cv.ORB_create(nfeatures = 10000, scaleFactor = 1.2, nlevels = 10, edgeThreshold = 31, firstLevel = 0,
                        WTA_K = 2, scoreType = 0, patchSize = 31, fastThreshold = 20)

    importFolder = r"C:\Users\84307\Desktop\UNI\Year 3\ELEC0036\Project\Project I\grayImages"
    exportFolder = r"C:\Users\84307\Desktop\UNI\Year 3\ELEC0036\Project\Project I\featureDetection"
    files = [file for file in os.listdir(importFolder) if os.path.isfile(os.path.join(importFolder, file))]
    keypointsList = []
    descriptorsList = []
    featureFrames = []

    for file in files:
        grayImage = cv.imread(os.path.join(importFolder, file))
        (keypoints, descriptors) = orb.detectAndCompute(grayImage, None)
        keypointsList.append(keypoints)
        descriptorsList.append(descriptors)
        frame = cv.drawKeypoints(grayImage, keypoints, None, color=(0,255,0))
        featureFrames.append(frame)
        path = os.path.join(exportFolder, file)
        cv.imwrite(path, frame)

    keypointsList = np.array(keypointsList, dtype=object)
    descriptorsList = np.array(descriptorsList, dtype=object)

    return keypointsList, descriptorsList


def featureMatch(keypointsList, descriptorsList):
    W, H, F, K = getCameraParams()
    importFolder_img = r"C:\Users\84307\Desktop\UNI\Year 3\ELEC0036\Project\Project I\importImages"
    files_img = [file for file in os.listdir(importFolder_img) if os.path.isfile(os.path.join(importFolder_img, file))]
    imagesList = []

    for file in files_img:
        path = os.path.join(importFolder_img, file)
        images = cv.imread(path)
        imagesList.append(images)

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)
    flann = cv.FlannBasedMatcher(index_params, search_params)

    src_pts_list = []
    dst_pts_list = []
    matches_list = []
    for i in range(len(descriptorsList) - 1):
        keypoints1 = keypointsList[i]
        keypoints2 = keypointsList[i + 1]
        descriptors1 = np.float32(descriptorsList[i])
        descriptors2 = np.float32(descriptorsList[i + 1])
        matches = flann.knnMatch(descriptors1, descriptors2, k=2)

        good = []
        for m, n in matches:
            if m.distance < 0.6 * n.distance:
                good.append(m)
        matches_list.append(np.array(good))

        src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good]).reshape(-1,2)
        src_pts_list.append(src_pts)
        dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good]).reshape(-1,2)
        dst_pts_list.append(dst_pts)
        M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()
        points = src_pts[mask.ravel() == 1]
        points2 = dst_pts[mask.ravel() == 1]

        draw_params = dict(matchColor = (0,255,0),
                           singlePointColor = None,
                           matchesMask = matchesMask,
                           flags = 2)

        result = cv.drawMatches(imagesList[i], keypointsList[i], imagesList[i + 1], keypointsList[i + 1], good, None, **draw_params)
        cv.imwrite(r"C:\Users\84307\Desktop\UNI\Year 3\ELEC0036\Project\Project I\matchResults\result_{}_{}.png".format(i + 1, i + 2), result)

        for (kp1, kp2) in zip(points, points2):
            (x1, y1) = int(kp1[0]), int(kp1[1])
            (x2, y2) = int(kp2[0]), int(kp2[1])
            pose = cv.line(imagesList[i], (x1,y1), (x2,y2), color=(255,0,0))
        cv.imwrite(r"C:\Users\84307\Desktop\UNI\Year 3\ELEC0036\Project\Project I\poseResults\pose_{}_{}.png".format(i + 1, i + 2), pose)

    matches_list = np.array(matches_list, dtype=object)

    return src_pts_list, dst_pts_list, matches_list


def poseEstimate(keypointsList, src_pts_list, dst_pts_list, matches_list):
    W, H, F, K = getCameraParams()
    K_32 = np.float32(K)

    structure = []

    src_pts = src_pts_list[0]
    dst_pts = dst_pts_list[0]
    P1 = []
    P2 = []
    E, mask_ = cv.findEssentialMat(src_pts, dst_pts, K)
    _, R, T, mask = cv.recoverPose(E, src_pts, dst_pts, K, mask_)
    for a in range(len(mask)):
        if mask[a] > 0:
            P1.append(src_pts[a])
    for b in range(len(mask)):
        if mask[b] > 0:
            P2.append(dst_pts[b])
    P1 = np.array(P1)
    P2 = np.array(P2)

    R0 = np.eye(3, 3)
    T0 = np.zeros((3, 1))

    projection1 = np.zeros((3, 4))
    projection2 = np.zeros((3, 4))
    projection1[0:3, 0:3] = np.float32(R0)
    projection1[:, 3] = np.float32(T0.T)
    projection2[0:3, 0:3] = np.float32(R)
    projection2[:, 3] = np.float32(T.T)
    projection1 = np.dot(K_32, projection1)
    projection2 = np.dot(K_32, projection2)
    point = cv.triangulatePoints(projection1, projection2, P1.T, P2.T)
    for i in range(len(point[0])):
        points3D = point[:, i]
        points3D /= points3D[3]
        structure.append([points3D[0], points3D[1], points3D[2]])

    structure = np.array(structure)
    rotations = [R0, R]
    motions = [T0, T]
    structureIndexList = []
    for keypoints in keypointsList:
        structureIndexList.append(np.ones(len(keypoints)) * - 1)
    structureIndexList = np.array(structureIndexList, dtype=object)
    index = 0
    matches = matches_list[0]
    for j, match in enumerate(matches):
        if mask[j] == 0:
            continue
        structureIndexList[0][int(match.queryIdx)] = index
        structureIndexList[1][int(match.trainIdx)] = index
        index += 1

    return rotations, motions, structure, structureIndexList


def combineStructure(matches, structureIndex, next_structureIndex, structure, next_structure):
    for k, match in enumerate(matches):
        query_idx = match.queryIdx
        train_idx = match.trainIdx
        struct_idx = structureIndex[query_idx]
        if struct_idx >= 0:
            next_structureIndex[train_idx] = struct_idx
            continue
        structure = np.append(structure, [next_structure[k]], axis=0)
        structureIndex[query_idx] = next_structureIndex[train_idx] = len(structure) - 1

    return structureIndex, next_structureIndex, structure


def pointCloud(rotations, motions, keypointsList, src_pts_list, dst_pts_list, matches_list, structure, structureIndexList):
    W, H, F, K = getCameraParams()
    K_32 = np.float32(K)
    arr = np.array([])
    object_points_list = []
    image_points_list = []
    structure_list = []
    structure_list.append(structure)

    for i in range(1, len(matches_list)):
        object_points = []
        image_points = []
        next_structure = []
        keypoints = keypointsList[i + 1]
        P1 = src_pts_list[i]
        P2 = dst_pts_list[i]
        matches = matches_list[i]
        structureIndex = structureIndexList[i]
        for match in matches:
            queryIdx = match.queryIdx
            trainIdx = match.trainIdx
            struct_idx = structureIndex[queryIdx]
            if struct_idx >= 0:
                object_points.append(structure[int(struct_idx)])
                image_points.append(keypoints[trainIdx].pt)
        object_points_list.append(np.array(object_points))
        image_points_list.append(np.array(image_points))

        retval, r, t, inliers = cv.solvePnPRansac(object_points_list[i - 1], image_points_list[i - 1], K_32, arr)
        R, J = cv.Rodrigues(r)
        rotations.append(R)
        motions.append(t)

        projection1 = np.zeros((3, 4))
        projection2 = np.zeros((3, 4))
        projection1[0:3, 0:3] = np.float32(rotations[i])
        projection1[:, 3] = np.float32(motions[i].T)
        projection2[0:3, 0:3] = np.float32(R)
        projection2[:, 3] = np.float32(t.T)
        projection1 = np.dot(K_32, projection1)
        projection2 = np.dot(K_32, projection2)
        point = cv.triangulatePoints(projection1, projection2, P1.T, P2.T)
        for j in range(len(point[0])):
            points3D = point[:, j]
            points3D /= points3D[3]
            next_structure.append([points3D[0], points3D[1], points3D[2]])
        structure_list.append(next_structure)

        structureIndexList[i], structureIndexList[i + 1], structure = \
            combineStructure(matches_list[i], structureIndexList[i],
                             structureIndexList[i + 1], structure, next_structure)

    for a in range(len(rotations)):
        r, _ = cv.Rodrigues(rotations[a])
        rotations[a] = r
    for b in range(len(structureIndexList)):
        point3dIndexList = structureIndexList[b]
        keypoints = keypointsList[b]
        r = rotations[b]
        t = motions[b]
        for c in range(len(point3dIndexList)):
            point3dIndex = int(point3dIndexList[c])
            if point3dIndex < 0:
                continue
            P, J = cv.projectPoints(structure[point3dIndex].reshape(1, 1, 3), r, t, K_32, arr)
            P = P.reshape(2)
            E = keypoints[c].pt - P
            if abs(E[0]) > 0.5 or abs(E[1]) > 0.5:
                new_point = None
            new_point = structure[point3dIndex]
            structure[point3dIndex] = new_point

    pointCloud = o3d.geometry.PointCloud()
    pointCloud.points = o3d.utility.Vector3dVector(structure[:,0:3])
    o3d.io.write_point_cloud(r"C:\Users\84307\Desktop\UNI\Year 3\ELEC0036\Project\Project I\pointCloud\pointCloud.pcd", pointCloud)
    mlab.points3d(structure[:, 0], structure[:, 1], structure[:, 2], mode='point', name='dinosaur')
    mlab.show()


def main():
    importFrames()

    (keypointsList, descriptorsList) = ORBDetect()
    (src_pts_list, dst_pts_list, matches_list) = featureMatch(keypointsList, descriptorsList)
    (rotations, motions, structure, structureIndexList) = poseEstimate(keypointsList, src_pts_list, dst_pts_list, matches_list)
    pointCloud(rotations, motions, keypointsList, src_pts_list, dst_pts_list, matches_list, structure, structureIndexList)

if __name__ == "__main__":
    main()