import os
from moviepy.video.io.VideoFileClip import VideoFileClip
from utils.voice_extraction import extract_audio, separate_voice, calculate_zero_crossing_rate
from utils.lip_motion import capture_lips_motion
import pandas as pd
import numpy as np

def save_data(video_file, shift, frames_count, create_header=True):
    '''
    Trigger the necessary functions and save data in csv format
    :param video_file: path to video file
    :param shift: shift of the video
    :param frames_count: number of frames to be analyzed
    :param create_header: whether to create header in csv file
    '''

    # Checking if the frame_count does not exceed length of video
    clip = VideoFileClip(video_file)
    frames = int(clip.fps * clip.duration)

    if frames_count < frames:
        # Saving audio data
        extract_audio(video_file)
        separate_voice(video_file.replace('.mp4', '.wav'))
        zcr = calculate_zero_crossing_rate(video_file.replace('.mp4', '_voice.wav'),
                                           video_file, frames_count=frames_count)

        # Saving video data
        lips_sep = capture_lips_motion(video_file, frame_count=frames_count)

        video_name = os.path.basename(os.path.normpath(video_file))

        main_df = pd.DataFrame({'Frame number': np.arange(1, frames_count+1),
                                'Id': [video_name] * frames_count,
                                'Shift': [shift] * frames_count,
                                'Voice': zcr,
                                'Video': lips_sep})
        main_df.to_csv("data/all_data.csv", mode='a', header=create_header, index=False)

    else:
        print(f'Video has less than {frames_count} frames. Aborting...')
