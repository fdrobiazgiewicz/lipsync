import math
import cv2
import dlib
import numpy as np


def capture_lips_motion(video_file, frame_count):
    '''
    Capture motion of the mouth in every frame of video.
    :param video_file: path to video file
    :param frame_count: number of frames to be analized
    :return: list with normalized lips separation distances on every frame
    '''
    cap = cv2.VideoCapture(video_file)
    i = 0
    lips_separation = []
    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret == False:
            break
        frameId = int(round(cap.get(1)))

        detected_face = face_detection(frame)
        landmarks = landmark_detection(detected_face, frame)
        print('\nframe number ', frameId)
        print('lips distance ', calculate_lips_distance(landmarks))
        lips_separation.append(calculate_lips_distance(landmarks))

        # Properties for visualization
        font = cv2.FONT_HERSHEY_SIMPLEX
        org = (10, 30)
        fontScale = 1
        color = (0, 0, 255)
        thickness = 1

        # Uncomment to visualize mouth point and calculated distance
        # cv2.putText(frame, f'Lips separation: {format(calculate_lips_distance(landmarks), ".2f")}', org, font, fontScale, color, thickness, cv2.LINE_AA)
        # cv2.imwrite(f"images/frame_{frameId}.jpg", frame)

        i += 1
        if frameId > frame_count - 1:
            cap.release()
            cv2.destroyAllWindows()

    normalized_lips = normalize_lip_separation_values(lips_separation)
    return normalized_lips

def face_detection(gray_img):
    '''
    detect face using dlib
    :param gray_img: frame in grayscale
    :return: area with face
    '''
    detector = dlib.get_frontal_face_detector()
    faces = detector(gray_img)
    return faces

def normalize_lip_separation_values(lips_separation):
    '''
    Express lip separation distances by its ratio from minimum to maximum value
    :param lips_separation: list with lips separation distances
    :return: list with normalized lips separation distances
    '''
    max_value = max(lips_separation)
    min_value = min(lips_separation)
    normalized = [(i-min_value)/(max_value-min_value) for i in lips_separation]
    return normalized


def landmark_detection(faces, gray_img):
    '''
    Detect facial landmarks.
    :param faces: area with face(s)
    :param gray_img: frame in grayscale
    :return: array with coordinates of mouth points
    '''
    landmark_detector = dlib.shape_predictor("utils/cascade_files/shape_predictor_68_face_landmarks.dat")
    for face in faces:
        landmarks = landmark_detector(gray_img,face)
        mouth_points = []
        # Get only (61, 68) landmarks to save the time
        for n in range(61, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            if n in [61, 62, 63, 65, 66, 67]:
                pass
                # Uncomment to visualize points
                # cv2.circle(gray_img, (x, y), 2, (0, 255, 0), -1)
            mouth_points.append([x, y])
            mouth_points_array = np.array(mouth_points) # Creating an array of coordinates of the landmarks.
    return mouth_points_array

def calculate_lips_distance(landmarks):
    '''
    Calculate the mean of mouth point distances.
    :param landmarks: array with mouth points cooridnates
    :return: value of lips separation
    '''

    A = distance(landmarks[0], landmarks[6])
    B = distance(landmarks[1], landmarks[5])
    C = distance(landmarks[2], landmarks[4])

    lips_separation_distance = (A + B + C) / 3.0

    return lips_separation_distance

def distance(p1, p2):
    '''
    Calculate distance between two points.
    :param p1: Point 1 cooridantes
    :param p2: Point 2 cooridantes
    :return: Distance between p1 and p2
    '''
    p1_x = p1[0]
    p2_x = p2[0]
    p1_y = p1[1]
    p2_y = p2[1]
    dist = math.sqrt((p2_x - p1_x) ** 2 + (p2_y - p1_y) ** 2)
    return dist
