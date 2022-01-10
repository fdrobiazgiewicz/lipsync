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
ap.add_argument(
    "-i",
    "--input",
    required=True,
    help="path to .txt with YouTube links"
)

ap.add_argument(
    "-f",
    "--frames",
    required=True,
    help="number of frames to analyse"
)

ap.add_argument(
    "-a",
    "--append",
    help="do not create header (appending existing .csv)",
    action='store_true'
)

args = vars(ap.parse_args())


if __name__ == "__main__":
    with open(args['input']) as file:
        lines = file.readlines()
        links = [line.rstrip() for line in lines]

    for link in links:
        id = video_id(link)
        pipeline(link, id)
        aligned_videos = os.listdir(f"db/{id}/aligned")
        print('aligned videos: ', aligned_videos)
        if len(aligned_videos) > 0:
            print(f'Found {len(aligned_videos)} aligned videos. Shifting...')
            for video, counter in zip(aligned_videos, range(1, len(aligned_videos))):
                face_file = f'db/{id}/aligned/{id}-{counter:03d}.mp4-aligned.avi'
                sequence_file = f'db/{id}/sequences/{id}-{counter:03d}.mp4'

                try:

                    clip = VideoFileClip(f"db/{id}/aligned/{video}")

                    # Skipping videos that failed during processing
                    if clip.duration > 6:
                        # Shifting for the random value from range (-1000, 1000) milliseconds
                        delta_millis = n = random.randint(-1000, 1000)
                        shift(face_path=face_file,
                              original_path=sequence_file,
                              video_id=id,
                              num_sequence=f'{counter:03d}',
                              delta_millis=delta_millis)

                        # Tracking voice and lips movement and saving it to .csv
                        shifted_path = f"db/{id}/shifted/{id}-{counter:03d}_shifted_" \
                                       f"{'plus' if delta_millis > 0 else 'minus'}_" \
                                       f"{abs(delta_millis)}.mp4"
                        if counter == 1 and not args['append']:
                            save_data(shifted_path, delta_millis, int(args['frames']))
                        else:
                            save_data(shifted_path, delta_millis, int(args['frames']), False)

                except OSError:
                    print('Corrupted file. Skipping...')

                except Exception as e:
                    print('Error: ', e)



