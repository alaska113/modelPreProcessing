
import importlib
import dlib
import cv2
from imutils import face_utils
import matplotlib.pyplot as plt
import numpy as np
import math
from statistics import mean

def extractFeaturesFromImage(imagePath):
    shapes, image, grayImage = getFacialLandmarks(cv2.imread(imagePath))
    yaw, pitch, roll = getHeadPosition(shapes[0], (426,640))
    croppedRightEyeGray = getRightEye(grayImage, shapes[0])
    croppedLeftEyeGray = getLeftEye(grayImage, shapes[0])
    return (yaw,pitch,roll), croppedLeftEyeGray, croppedRightEyeGray

def getFacialLandmarks(image):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    grayImage = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    rects = detector(grayImage)
    shapes = []

    for rect in rects:
        shape = predictor(grayImage,rect)
        shape = face_utils.shape_to_np(shape)
        shapes.append(shape)
    return shapes, image, grayImage

def getRightEye(grayImage, landmarks):
    desiredSize = [60, 36]
    middlePoint = (mean([landmarks[39][0], landmarks[36][0]]), mean([landmarks[39][1], landmarks[36][1]]))
    upLeft = (int(middlePoint[0] - desiredSize[0]/2), int(middlePoint[1] + desiredSize[1]/2))
    botRight = (int(middlePoint[0] + desiredSize[0]/2), int(middlePoint[1] - desiredSize[1]/2))
    croppedEyeGray = grayImage[botRight[1]:upLeft[1], upLeft[0]:botRight[0]]
    return croppedEyeGray

def getLeftEye(grayImage, landmarks):
    desiredSize = [60, 36]
    middlePoint = (mean([landmarks[42][0], landmarks[45][0]]), mean([landmarks[42][1], landmarks[45][1]]))
    upLeft = (int(middlePoint[0] - desiredSize[0]/2), int(middlePoint[1] + desiredSize[1]/2))
    botRight = (int(middlePoint[0] + desiredSize[0]/2), int(middlePoint[1] - desiredSize[1]/2))
    croppedEyeGray = grayImage[botRight[1]:upLeft[1], upLeft[0]:botRight[0]]
    return croppedEyeGray


#720x480
def getHeadPosition(landmarksArr, resolution):
    landmarks = np.array(
        [
            (landmarksArr[45][0], landmarksArr[45][1]),
            (landmarksArr[36][0], landmarksArr[36][1]),
            (landmarksArr[33][0], landmarksArr[33][1]),
            (landmarksArr[54][0], landmarksArr[54][1]),
            (landmarksArr[48][0], landmarksArr[48][1]),
            (landmarksArr[8][0],  landmarksArr[8][1]),
        ], dtype=np.float,
    )
    image_points = np.array(
        [
            (landmarks[2][0], landmarks[2][1]),
            (landmarks[5][0], landmarks[5][1]),
            (landmarks[0][0], landmarks[0][1]),
            (landmarks[1][0], landmarks[1][1]),
            (landmarks[3][0], landmarks[3][1]),
            (landmarks[4][0], landmarks[4][1]),
        ], dtype=np.float,
    )
    for point in image_points:
        point = rotate_landmark(image_points[0], point, np.pi)

    model_points = np.array(
        [
            (0.0, 0.0, 0.0), (0.0, -330.0, -65.0), (-165.0, 170.0, -135.0),
            (165.0, 170.0, -135.0), (
                -150.0, -
                150.0, -125.0,
            ), (150.0, -150.0, -125.0),
        ], dtype=np.float,
    )
    # Camera internals
    center = (resolution[1]/2, resolution[0]/2)
    focal_length = center[0] / np.tan((60.0/2.0) * (np.pi / 180.0))
    camera_matrix = np.array(
        [
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1],
        ], dtype=np.float,
    )
    # Assuming no lens distortion
    dist_coeffs = np.zeros((4, 1), dtype="double",)

    success, rotation_vector, translation_vector = cv2.solvePnP(
        model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE,
    )

    axis = np.float32([
        [500, 0, 0],
        [0, 500, 0],
        [0, 0, 500],
    ])

    imgpts, jac = cv2.projectPoints(
        axis, rotation_vector, translation_vector, camera_matrix, dist_coeffs,
    )
    modelpts, jac2 = cv2.projectPoints(
        model_points, rotation_vector, translation_vector, camera_matrix, dist_coeffs,
    )
    rvec_matrix = cv2.Rodrigues(rotation_vector)[0]

    proj_matrix = np.hstack((rvec_matrix, translation_vector))
    eulerAngles = cv2.decomposeProjectionMatrix(proj_matrix)[6]

    pitch, yaw, roll = [math.radians(_) for _ in eulerAngles]

    pitch = math.degrees(math.asin(math.sin(pitch)))
    roll = -math.degrees(math.asin(math.sin(roll)))
    yaw = math.degrees(math.asin(math.sin(yaw)))

    return yaw, pitch, roll


def rotate_landmark(origin, point, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.

    The angle should be given in radians.
    """
    ox, oy = origin
    px, py = point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return qx, qy

def normalize(grayscale):
    return (grayscale - min(grayscale))/(max(grayscale)-min(grayscale))


if __name__ == "__main__":
    imagePath = '/Users/khan/code/attently/eyesCenter.jpg'
    (yaw, pitch, roll), leftEye, rightEye = extractFeaturesFromImage(imagePath)

    flattenedLeftEye = np.array(leftEye, dtype=np.float32).flatten()
    normalizedLeft = np.resize(np.array(normalize(flattenedLeftEye)), (36,60))

    flattenedRightEye = np.array(rightEye, dtype=np.float32).flatten()
    normalizedRight = np.resize(np.array(normalize(flattenedRightEye)), (36,60))
