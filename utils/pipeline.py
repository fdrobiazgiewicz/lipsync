from os import listdir
import sys
from os.path import basename
from pathlib import Path

from utils.analyze_shots import analyze_shots
from utils.analyze_shots import extract_sequences
from utils.downloader import download
from utils.extract_align_faces import process_with_buffered_reader
from pytube.extract import video_id


def analyze_and_cut(stream, file_path):
    print(f'downloaded from stream: {stream} to {file_path}')
    file_name = basename(file_path).split('.')[0]
    applicable_scenes = analyze_shots(video_path=file_path, peek_frames=False)
    extract_sequences(db_path='./db/', filename=file_name, scene_list=applicable_scenes)
    # TODO: delete the original to preserve disk space?

def pipeline(URL, ID):
    download(video_url=URL,
             destination_path='./db/raw',
             on_callback=analyze_and_cut,
             attempts=3)

    sequences_path = f'./db/{ID}/sequences/'
    aligned_dir = f'./db/{ID}/aligned/'
    Path(aligned_dir).mkdir(parents=True, exist_ok=True)
    for sequence in listdir(sequences_path):
        print(f'processing {sequence}...')
        try:
            process_with_buffered_reader(path=sequences_path + sequence,
                                         output_name=aligned_dir + f'{sequence}-aligned.avi',
                                         desired_width=256)
        except AssertionError:
            print(f'...error occurred processing {sequence}, aborting')
            continue
        print(f'...done processing {sequence}')



if __name__ == '__main__':
    URL = sys.argv[1]

    ID = video_id(URL)

    pipeline(URL, ID)
