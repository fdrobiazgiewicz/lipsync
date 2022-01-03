import argparse
import math
import cv2
import dlib
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt

# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--input", required=True,
#                 help="path to input video folder")
# # ap.add_argument("-o", "--output", required=True,
# #                 help="path to output csv file")
# args = vars(ap.parse_args())


def capture_lips_motion(video_file, frame_count):
    cap = cv2.VideoCapture(video_file)
    framespersecond = int(cap.get(cv2.CAP_PROP_FPS))
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
        # font
        font = cv2.FONT_HERSHEY_SIMPLEX

        # org
        org = (10, 20)

        # fontScale
        fontScale = 0.4

        # Blue color in BGR
        color = (0, 0, 255)

        # Line thickness of 2 px
        thickness = 1

       # cv2.putText(frame, f'Lips separation: {format(calculate_lips_distance(landmarks), ".2f")}', org, font, fontScale, color, thickness, cv2.LINE_AA)
       # cv2.imwrite(f"images/frame_{frameId}.jpg", frame)

        i += 1
        if frameId > frame_count - 1:
            cap.release()
            cv2.destroyAllWindows()

    print("Lips separation normalized:\n ", normalize_lip_separation_values(lips_separation))
    normalized_lips = normalize_lip_separation_values(lips_separation)
    # plt.plot(normalized_lips)
    # plt.show()
    return normalized_lips

def face_detection(gray_img):
    detector = dlib.get_frontal_face_detector()
    faces = detector(gray_img)
    return faces

def normalize_lip_separation_values(lips_separation):
    max_value = max(lips_separation)
    min_value = min(lips_separation)
    normalized = [(i-min_value)/(max_value-min_value) for i in lips_separation]
    return normalized

def get_shift_from_video_name(video_path):
    video_name = video_path.split("\\")[-1]
    shift_part = video_name.split('shifted_')[-1]
    if 'plus' in shift_part:
        shift_value = shift_part.replace('.mp4', '')
        shift_value = int(shift_value.replace('plus_', ''))
    else:
        shift_value = shift_part.replace('.mp4', '')
        shift_value = int(shift_value.replace('minus_', '-'))
    return shift_value


def landmark_detection(faces,gray_img):
    landmark_detector = dlib.shape_predictor("utils/cascade_files/shape_predictor_68_face_landmarks.dat")
    for face in faces:
        landmarks = landmark_detector(gray_img,face)
        mouth_points = []
        # Get only (61, 68) landmarks to save the time
        for n in range(61, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            if n in [61, 62, 63, 65, 66, 67]:
                cv2.circle(gray_img, (x, y), 2, (0, 255, 0), -1)
            mouth_points.append([x,y])
            mouth_points_array = np.array(mouth_points) # Creating an array of coordinates of the landmarks.
            # The above two lines can be used to display the landmarks and get the indices of other parts like nose,eyes etc.
    return mouth_points_array

def calculate_lips_distance(landmarks):

    A = distance(landmarks[0], landmarks[6])
    B = distance(landmarks[1], landmarks[5])
    C = distance(landmarks[2], landmarks[4])

    lips_separation_distance = (A + B + C) / 3.0

    return lips_separation_distance

def distance(p1, p2):
    p1_x = p1[0]
    p2_x = p2[0]
    p1_y = p1[1]
    p2_y = p2[1]
    dist = math.sqrt((p2_x - p1_x) ** 2 + (p2_y - p1_y) ** 2)
    return dist



if __name__ == "__main__":
    # capture_lips_motion(args['input'])
    pass
