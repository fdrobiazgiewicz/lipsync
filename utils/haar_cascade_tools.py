import cv2
import numpy as np

MIN_SIZE = 50
SCALE_FACTOR = 1.3
HAAR_CASCADE_FILE = 'haarcascade_frontalface_default.xml'
CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + HAAR_CASCADE_FILE)


def has_one_face(file, show_result=False, min_neighbors=18):
    gray = cv2.cvtColor(cv2.imread(file), cv2.COLOR_BGR2GRAY)

    print(f'running simple haar cascade face detection for {file}...')
    faces = CASCADE.detectMultiScale(gray,
                                     scaleFactor=SCALE_FACTOR,
                                     minNeighbors=min_neighbors,
                                     minSize=(MIN_SIZE, MIN_SIZE))
    print(f'...done, found {len(faces)}')
    if show_result:
        mark_show_detections(gray, faces)

    return len(faces) == 1


def detect_with_fallback(frame_mat,
                         show_result=False,
                         output_size=256,
                         fallback_frame=None):
    try:
        return detect_and_scale(frame_mat, show_result=show_result, output_size=output_size)
    except AssertionError:
        return fallback_frame or np.zeros((output_size, output_size), dtype='uint8')


def detect_and_scale(frame_mat, show_result=False, output_size=256):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'utils/cascade_files/haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'utils/cascade_files/haarcascade_eye.xml')

    gray = cv2.cvtColor(frame_mat, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,
                                          scaleFactor=1.3,
                                          minNeighbors=18,
                                          minSize=(50, 50))
    assert len(faces) == 1

    if show_result:
        mark_show_detections(gray, faces)

    face_roi = trim(gray, faces[0])

    eyes = eye_cascade.detectMultiScale(face_roi,
                                        scaleFactor=1.3,
                                        minNeighbors=18)
    if show_result:
        mark_show_detections(face_roi, eyes, 'first faces')
    assert len(eyes) == 2

    e1, e2 = eyes
    (left_eye_rect, right_eye_rect) = (e1, e2) if e1[0] < e2[0] else (e2, e1)

    left_eye = rectangle_center(left_eye_rect)
    right_eye = rectangle_center(right_eye_rect)

    dx = left_eye[0] - right_eye[0]
    dy = left_eye[1] - right_eye[1]
    angle = np.arctan(dy / dx) * 180 / np.pi

    im_height, im_width = gray.shape
    image_center = im_height // 2, im_width // 2
    # image_center = (left_eye[0] + right_eye[0]) // 2, (left_eye[1] + right_eye[1]) // 2
    rotation_matrix = cv2.getRotationMatrix2D(center=image_center, angle=angle, scale=1.0)
    rotated = cv2.warpAffine(src=gray,
                             M=rotation_matrix,
                             dsize=(im_width, im_height))
    if show_result:
        cv2.imshow('rotated', rotated)
        cv2.waitKey()

    straightened_faces = face_cascade.detectMultiScale(rotated,
                                                       scaleFactor=1.3,
                                                       minNeighbors=18,
                                                       minSize=(50, 50))
    assert len(straightened_faces) == 1

    if show_result:
        mark_show_detections(rotated, straightened_faces, 'straightened')

    cropped = trim(rotated, straightened_faces[0], padding=15)

    resized = cv2.resize(cropped,
                         dsize=(output_size, output_size),
                         interpolation=cv2.INTER_CUBIC)
    return resized


def trim(src, rectangle, padding=0):
    (x, y, h, w) = rectangle
    if padding == 0:
        return src[y:y + h, x:x + w]
    else:
        return src[y - padding + 1:y + h + padding, x - padding + 1:x + w + padding]


def mark_show_detections(src, detections, title='detections'):
    tmp = np.copy(src)  # that's how you do immutability in python
    for x, y, h, w in detections:
        cv2.rectangle(tmp, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.imshow(title, tmp)
    cv2.waitKey()


def rectangle_center(rectangle):
    top_x, top_y, height, width = rectangle
    return top_x + width // 2, top_y + height // 2


if __name__ == '__main__':
    DB_PATH = '/Users/phili/private_workspace/engineer/db/'
    frame_file = 'dP-IlGZBH1E/pictures/002-01.jpg'
    PATH = DB_PATH + frame_file

    out = detect_with_fallback(cv2.imread(PATH),
                               output_size=256,
                               show_result=True)
    cv2.imshow('result', out)
    cv2.waitKey()
