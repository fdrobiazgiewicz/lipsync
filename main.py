import argparse
import os
import sys
import random

from utils.pipeline import pipeline
from pytube.extract import video_id
from utils.shifting import shift
from utils.data_preparation import save_data
from moviepy.editor import VideoFileClip

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
                help="path to .txt with YouTube links")
ap.add_argument("-a", "--append",
                 help="do not create header (appending existing .csv)")
args = vars(ap.parse_args())



if __name__ == "__main__":
    with open('D:/links.txt') as file:
        lines = file.readlines()
        links = [line.rstrip() for line in lines]

    for link in links:
        id = video_id(link)
        pipeline(link, id)
        aligned_videos = os.listdir(f"db/{id}/aligned")
        if len(aligned_videos) > 0:
            print(f'Found {len(aligned_videos)} aligned videos. Shifting...')
            sequence = '001'
            for video, counter in zip(aligned_videos, range(1, len(aligned_videos))):
                sequence = f'00{counter}'
                face_file = f'db/{id}/aligned/{id}-{sequence}.mp4-aligned.avi'
                sequence_file = f'db/{id}/sequences/{id}-{sequence}.mp4'

                clip = VideoFileClip(f"db/{id}/aligned/{video}")

                # Skipping videos that failed during processing
                if clip.duration > 6:
                    # Shifting for the random value from range (-1000, 1000) milliseconds
                    delta_millis = n = random.randint(-1000, 1000)
                    shift(face_path=face_file,
                          original_path=sequence_file,
                          video_id=id,
                          num_sequence=sequence,
                          delta_millis=delta_millis)

                    # Tracking voice and lips movement and saving it to .csv
                    shifted_path = f"db/{id}/shifted/{id}-{sequence}_shifted_{'plus' if delta_millis > 0 else 'minus'}_{abs(delta_millis)}.mp4"
                    save_data(shifted_path, delta_millis)



