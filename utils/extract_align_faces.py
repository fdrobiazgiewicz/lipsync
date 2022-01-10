import os
import cv2
from dlib import get_frontal_face_detector, shape_predictor
from imutils.face_utils import facealigner
from imutils.video import FileVideoStream
from tqdm import tqdm

LANDMARKS_FILE_68 = os.getcwd() + '/utils/cascade_files/shape_predictor_68_face_landmarks.dat'
LANDMARKS_FILE_5 = os.getcwd() + '/utils/cascade_files/shape_predictor_5_face_landmarks.dat'
WIDTH = 256

FACE_DETECTOR = get_frontal_face_detector()
PREDICTOR = shape_predictor(LANDMARKS_FILE_68)
# TODO: it's fixed width!
FACE_ALIGNER = facealigner.FaceAligner(predictor=PREDICTOR,
                                       desiredFaceWidth=WIDTH)


def extract_and_align_face(frame_mat,
                           show_result=False):
    gray = cv2.cvtColor(frame_mat, cv2.COLOR_BGR2GRAY)
    # upscale neither once nor twice!
    rectangles = FACE_DETECTOR(gray, 0)
    assert len(rectangles) == 1
    rectangle = rectangles[0]

    face = FACE_ALIGNER.align(frame_mat, gray, rectangle)
    if show_result:
        cv2.imshow('hog result', face)
        cv2.waitKey()

    return cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)


def process_with_buffered_reader(path, output_name, desired_width):
    frame_count, framerate = get_parameters_for_file(path)

    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    result = cv2.VideoWriter(output_name,
                             fourcc,
                             framerate,
                             (desired_width, desired_width),
                             isColor=False)

    # I think it's better to separate reading from processing. It seems a little bit faster.
    # Wouldn't be reasonable to separate writing as well?
    # It goes sequentially, however a short write buffer
    buffered_capture = FileVideoStream(path=path,
                                       queue_size=128)

    buffered_capture.start()
    progress_bar = tqdm(total=frame_count)
    while buffered_capture.more() and progress_bar.n < frame_count:
        aligned = extract_and_align_face(frame_mat=buffered_capture.read(),
                                         show_result=False)
        result.write(aligned)
        progress_bar.update()
    progress_bar.close()
    result.release()


def get_parameters_for_file(path):
    capture = cv2.VideoCapture(path)
    framerate = round(capture.get(cv2.CAP_PROP_FPS))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    return frame_count, framerate


if __name__ == '__main__':
    DB_PATH = '/Users/phili/private_workspace/engineer/db/'
    frame_file = 'dP-IlGZBH1E/pictures/002-01.jpg'
    PATH = DB_PATH + frame_file
    video_path = r'/Users/phili/private_workspace/engineer/db/raw/dP-IlGZBH1E.mp4'

    process_with_buffered_reader(video_path, 'processed.avi', desired_width=256)
