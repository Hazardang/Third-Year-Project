import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import open3d as o3d

def getCameraParams():
    W = 4032
    H = 3024
    F = 26
    K = np.array([[3385.4409,0,2024.6423],[0,3384.9633,1499.2148],[0,0,1]])
    return W, H, F, K


def undistort():
    return 0


def cart2hom(arr):
    if arr.ndim == 1:
        return np.hstack([arr, 1])
    return np.asarray(np.vstack([arr, np.ones(arr.shape[1])]))


def correspondence_matrix(p1, p2):
    p1x, p1y = p1[:2]
    p2x, p2y = p2[:2]

    return np.array([
        p1x * p2x, p1x * p2y, p1x,
        p1y * p2x, p1y * p2y, p1y,
        p2x, p2y, np.ones(len(p1x))
    ]).T

    return np.array([
        p2x * p1x, p2x * p1y, p2x,
        p2y * p1x, p2y * p1y, p2y,
        p1x, p1y, np.ones(len(p1x))
    ]).T


def scale_and_translate_points(points):
    x = points[0]
    y = points[1]
    center = points.mean(axis=1)  # mean of each row
    cx = x - center[0]  # center the points
    cy = y - center[1]
    dist = np.sqrt(np.power(cx, 2) + np.power(cy, 2))
    scale = np.sqrt(2) / dist.mean()
    norm3d = np.array([
        [scale, 0, -scale * center[0]],
        [0, scale, -scale * center[1]],
        [0, 0, 1]
    ])

    return np.dot(norm3d, points), norm3d


def compute_image_to_image_matrix(x1, x2, compute_essential=False):
    A = correspondence_matrix(x1, x2)
    # compute linear least square solution
    U, S, V = np.linalg.svd(A)
    F = V[-1].reshape(3, 3)

    # constrain F. Make rank 2 by zeroing out last singular value
    U, S, V = np.linalg.svd(F)
    S[-1] = 0
    if compute_essential:
        S = [1, 1, 0]  # Force rank 2 and equal eigenvalues
    F = np.dot(U, np.dot(np.diag(S), V))

    return F


def compute_normalized_image_to_image_matrix(p1, p2, compute_essential=False):
    n = p1.shape[1]
    if p2.shape[1] != n:
        raise ValueError('Number of points do not match.')

    # preprocess image coordinates
    p1n, T1 = scale_and_translate_points(p1)
    p2n, T2 = scale_and_translate_points(p2)

    # compute F or E with the coordinates
    F = compute_image_to_image_matrix(p1n, p2n, compute_essential)

    # reverse preprocessing of coordinates
    # We know that P1' E P2 = 0
    F = np.dot(T1.T, np.dot(F, T2))

    return F / F[2, 2]


def compute_P_from_essential(E):
    U, S, V = np.linalg.svd(E)

    # Ensure rotation matrix are right-handed with positive determinant
    if np.linalg.det(np.dot(U, V)) < 0:
        V = -V

    # create 4 possible camera matrices (Hartley p 258)
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    P2s = [np.vstack((np.dot(U, np.dot(W, V)).T, U[:, 2])).T,
           np.vstack((np.dot(U, np.dot(W, V)).T, -U[:, 2])).T,
           np.vstack((np.dot(U, np.dot(W.T, V)).T, U[:, 2])).T,
           np.vstack((np.dot(U, np.dot(W.T, V)).T, -U[:, 2])).T]

    return P2s


def compute_essential_normalized(p1, p2):
    return compute_normalized_image_to_image_matrix(p1, p2, compute_essential=True)


def skew(x):
    return np.array([
        [0, -x[2], x[1]],
        [x[2], 0, -x[0]],
        [-x[1], x[0], 0]
    ])


def reconstruct_one_point(pt1, pt2, m1, m2):
    A = np.vstack([
        np.dot(skew(pt1), m1),
        np.dot(skew(pt2), m2)
    ])
    U, S, V = np.linalg.svd(A)
    P = np.ravel(V[-1, :4])

    return P / P[3]


def linear_triangulation(p1, p2, m1, m2):
    num_points = p1.shape[1]
    res = np.ones((4, num_points))

    for i in range(num_points):
        A = np.asarray([
            (p1[0, i] * m1[2, :] - m1[0, :]),
            (p1[1, i] * m1[2, :] - m1[1, :]),
            (p2[0, i] * m2[2, :] - m2[0, :]),
            (p2[1, i] * m2[2, :] - m2[1, :])
        ])

        _, _, V = np.linalg.svd(A)
        X = V[-1, :4]
        res[:, i] = X / X[3]

    return res


def readImage():
    imageOriginal = cv.imread("image.png")
    image = cv.resize(imageOriginal, (4032,3024))
    imageOriginal2 = cv.imread("image2.png")
    image2 = cv.resize(imageOriginal2, (4032,3024))
    return image, image2

def cvtRGB2Gray(image, image2):
    grayImage = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    cv.imwrite('grayImage.png', grayImage)
    grayImage32 = np.float32(grayImage)
    grayImage2 = cv.cvtColor(image2, cv.COLOR_BGR2GRAY)
    cv.imwrite('grayImage2.png', grayImage2)
    grayImage32_2 = np.float32(grayImage2)
    return grayImage, grayImage32, grayImage2, grayImage32_2


def cornerDetect(image, image2, grayImage32, grayImage32_2):
    dst = cv.cornerHarris(grayImage32, blockSize=15, ksize=3, k=0.04)
    threshold = 80
    for i in range(dst.shape[0]):
        for j in range(dst.shape[1]):
            if int(dst[i,j]) > threshold:
                cv.circle(image, (j,i), 4, (0,255,0), 2)
                cv.circle(dst, (j,i), 4, (0,255,0), 2)
    cv.imwrite('image(CORNER).png', image)
    cv.imwrite('image(CORNERGRAY).png', dst)

    dst2 = cv.cornerHarris(grayImage32_2, blockSize=15, ksize=3, k=0.04)
    threshold = 80
    for i in range(dst2.shape[0]):
        for j in range(dst2.shape[1]):
            if int(dst2[i,j]) > threshold:
                cv.circle(image2, (j,i), 4, (0,255,0), 2)
                cv.circle(dst2, (j,i), 4, (0,255,0), 2)
    cv.imwrite('image2(CORNER).png', image2)
    cv.imwrite('image2(CORNERGRAY).png', dst2)

    #while (True):
    #    cv.imshow('Source Image', image2)
    #    cv.imshow('Corner Detection', dst2)
    #    if cv.waitKey(120) & 0xff == ord("q"):
    #        break


def fastFeatureDetect(image, image2):
    fast = cv.FastFeatureDetector_create()

    keypoints = fast.detect(image, None)
    dst = image.copy()
    dst = cv.drawKeypoints(image, keypoints, dst, color=(0,255,0))
    cv.imwrite('image(FAST).png', dst)

    keypoints2 = fast.detect(image2, None)
    dst2 = image2.copy()
    dst2 = cv.drawKeypoints(image2, keypoints2, dst2, color=(0,255,0))
    cv.imwrite('image2(FAST).png', dst2)

    #while (True):
    #    cv.imshow('Fast Keypoints Detection', dst)
    #    if cv.waitKey(120) & 0xff == ord("q"):
    #        break


def BRISKFeatureDetect(image, image2, grayImage32, grayImage32_2):
    brisk = cv.BRISK_create(thresh=90, octaves=3, patternScale=1.0)

    (keypoints, descriptor) = brisk.detectAndCompute(image, None)
    dst = image.copy()
    dst = cv.drawKeypoints(image, keypoints, dst, color=(0,255,0))
    cv.imwrite('image(BRISK).png', dst)

    (keypoints2, descriptor2) = brisk.detectAndCompute(image2, None)
    dst2 = image2.copy()
    dst2 = cv.drawKeypoints(image2, keypoints2, dst2, color=(0,255,0))
    cv.imwrite('image2(BRISK).png', dst2)

    matcher = cv.BFMatcher(normType = cv.NORM_HAMMING)
    matches = matcher.knnMatch(descriptor, descriptor2, k=2)
    matches = sorted(matches, key=lambda x:x[0].distance)
    good = []
    for m, n in matches:
        if m.distance < 0.75*n.distance:
            good.append([m])

    out_img = cv.drawMatchesKnn(img1=image, keypoints1=keypoints, img2=image2, keypoints2=keypoints2, matches1to2=good, outImg=None, flags=2)
    cv.imwrite('out_img(BRISK_BF).png', out_img)
    cv.imshow('out_img', out_img)

    #while (True):
    #    cv.imshow('BRISK keypoints Detection', dst)
    #    if cv.waitKey(120) & 0xff == ord("q"):
    #        break


def ORBDetect(image, grayImage, image2, grayImage2):
    orb = cv.ORB_create(nfeatures = 10000, scaleFactor = 1.2, nlevels = 10, edgeThreshold = 31, firstLevel = 0, WTA_K = 2, scoreType = 0, patchSize = 31, fastThreshold = 20)

    (keypoints, descriptor) = orb.detectAndCompute(grayImage, None)
    frame1 = image.copy()
    frame1 = cv.drawKeypoints(image, keypoints, frame1, color=(0,255,0))
    cv.imwrite('image(ORB).png', frame1)

    (keypoints2, descriptor2) = orb.detectAndCompute(grayImage2, None)
    frame2 = image2.copy()
    frame2 = cv.drawKeypoints(image2, keypoints2, frame2, color=(0,255,0))
    cv.imwrite('image2(ORB).png', frame2)

    descriptor_32 = np.float32(descriptor)
    descriptor2_32 = np.float32(descriptor2)

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)
    flann = cv.FlannBasedMatcher(index_params, search_params)
    #matcher = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
    matches = flann.knnMatch(descriptor_32, descriptor2_32, k=2)
    #matches = sorted(matches, key = lambda x:x.distance)

    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)

    MIN_MATCH_COUNT = 10
    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32([keypoints[m.queryIdx].pt for m in good]).reshape(-1,2)
        dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good]).reshape(-1,2)
        M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()
        points = src_pts[mask.ravel() == 1]
        points2 = dst_pts[mask.ravel() == 1]
        W, H, F, K = getCameraParams()
        pts = np.float32([ [0,0],[0,H-1],[W-1,H-1],[W-1,0] ]).reshape(-1,1,2)
        dst = cv.perspectiveTransform(pts, M)
    else:
        print("Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT) )
        matchesMask = None

    draw_params = dict(matchColor = (0,255,0),
                       singlePointColor = None,
                       matchesMask = matchesMask,
                       flags = 2)

    for kp1, kp2 in zip(points, points2):
        (x1, y1) = int(kp1[0]), int(kp1[1])
        (x2, y2) = int(kp2[0]), int(kp2[1])
        out_img_pose = cv.line(frame1, (x1,y1), (x2,y2), color=(255,0,0))
    cv.imwrite('out_img(ORB_POSE).png', out_img_pose)

    out_img = cv.drawMatches(image, keypoints, image2, keypoints2, good, None, **draw_params)
    cv.imwrite('out_img(ORB_RANSAC).png', out_img)
    plt.imshow(out_img, 'gray')
    plt.show()

    #out_img = cv.drawMatches(img1=image, keypoints1=keypoints, img2=image2, keypoints2=keypoints2, matches1to2=matches[:], outImg=None, flags=2)
    #cv.imwrite('out_img(ORB_BF).png', out_img)

    #src_pts = np.asarray([keypoints[m.queryIdx].pt for m in matches])
    #dst_pts = np.asarray([keypoints2[m.trainIdx].pt for m in matches])
    #M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 100.0)
    ##mask = mask.ravel()
    #pts = src_pts[mask.ravel() == 1]
    #pts2 = dst_pts[mask.ravel() == 1]

    #h, w = image.shape
    #pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
    #dst = cv.perspectiveTransform(pts, M)
    #image2 = cv.polylines(image2, [np.int32(dst)], True, (255, 255, 255), 3, cv.LINE.AA)

    #out_img_Ransac = cv.drawMatches(img1=image, keypoints1=pts, img2=image2, keypoints2=pts2, matches1to2=matches, outImg=None, flags=2)
    #cv.imwrite('out_img(ORB_BF_RANSAC).png', out_img_Ransac)

    return points, points2


def createPointCloud(image, image2, points, points2, K):
    pts = cart2hom(points.T)
    pts2 = cart2hom(points2.T)

    fig, axis = plt.subplots(1, 2)
    axis[0].autoscale_view('tight')
    axis[0].imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))
    axis[0].plot(pts[0], pts[1], 'r.')
    axis[1].autoscale_view('tight')
    axis[1].imshow(cv.cvtColor(image2, cv.COLOR_BGR2RGB))
    axis[1].plot(pts2[0], pts2[1], 'r.')
    plt.savefig('matchesPoints(ORB_BF).png')

    points1n = np.dot(np.linalg.inv(K), pts)
    points2n = np.dot(np.linalg.inv(K), pts2)
    E = compute_essential_normalized(points1n, points2n)

    P1 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
    P2s = compute_P_from_essential(E)

    ind = -1
    for i, P2 in enumerate(P2s):
        d1 = reconstruct_one_point(points1n[:, 0], points2n[:, 0], P1, P2)
        P2_homogenous = np.linalg.inv(np.vstack([P2, [0, 0, 0, 1]]))
        d2 = np.dot(P2_homogenous[:3, :4], d1)
        if d1[2] > 0 and d2[2] > 0:
            ind = i

    P2 = np.linalg.inv(np.vstack([P2s[ind], [0, 0, 0, 1]]))[:3, :4]
    Points4D = linear_triangulation(points1n, points2n, P1, P2)

    Points4D = -1 * Points4D
    #Points4D = Points4D.T
    #print(Points4D.shape)
    #Points3D = np.delete(Points4D,3,axis=1)
    #PointCloud = o3d.geometry.PointCloud()
    #PointCloud.points = o3d.utility.Vector3dVector(Points3D)
    #o3d.visualization.draw_geometries([PointCloud])

    fig = plt.figure()
    fig.suptitle('3D Point Cloud', fontsize=16)
    ax = fig.add_subplot(projection='3d')
    ax.plot(Points4D[0], Points4D[1], Points4D[2], 'b.')
    ax.set_xlabel('x axis')
    ax.set_ylabel('y axis')
    ax.set_zlabel('z axis')
    ax.view_init(elev=135, azim=90)
    plt.savefig('3DPointCloud.png')
    plt.show()


#def StereoDisparity(grayImage, grayImage2):
#    stereo = cv.StereoBM_create(numDisparities=16, blockSize=5)
#    disparity = stereo.compute(grayImage, grayImage2)
#    plt.imshow(disparity, 'gray')
#    plt.show()

#def SIFTDetect(grayImage32, image):
#    sift = cv.SIFT_create()
#    keypoints = sift.detect(grayImage32, None)
#    dst = cv.drawKeypoints(grayImage32, keypoints, image)
#    cv.imwrite('image(SIFT).png', image)
#    cv.imwrite('image(SIFTGRAY).png', dst)


#def SURFDetect(grayImage32, image):
#    surf = cv.SURF_create()
#    keypoints = surf.detect(grayImage32, None)
#    dst = cv.drawKeypoints(grayImage32, keypoints, image)
#    cv.imwrite('image(SURF).png', image)
#    cv.imwrite('image(SURFGRAY).png', dst)


#def showCamera():
#    frameWidth = 1920
#    frameHeight = 1080
#    capture = cv.VideoCapture(1)
#    capture.set(1, frameWidth)
#    capture.set(1, frameHeight)
#    capture.set(1, 150)
#    while True:
#        success, image = capture.read()
#        cv.imshow("Result", image)
#        if cv.waitKey(1) & 0xFF == ord('q'):
#            break


def main():
    W, H, F, K = getCameraParams()

    (image, image2) = readImage()

    (grayImage, grayImage32, grayImage2, grayImage32_2) = cvtRGB2Gray(image, image2)

    #(keypoints, descriptor, keypoints2, descriptor2) = ORBDetect(image, grayImage, image2, grayImage2)

    (points, points2) = ORBDetect(image, grayImage, image2, grayImage2)

    undistort()

    #cornerDetect(image, image2, grayImage32, grayImage32_2)
    #fastFeatureDetect(image, image2)
    #BRISKFeatureDetect(image, image2, grayImage32, grayImage32_2)
    #SIFTDetect(grayImage32, image)
    #SURFDetect(grayImage32, image)
    #ORBDetect(image, grayImage, image2, grayImage2)

    createPointCloud(image, image2, points, points2, K)

    #StereoDisparity(grayImage, grayImage2)

if __name__ == "__main__":
    main()
