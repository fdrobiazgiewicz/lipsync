import subprocess
from os import listdir
from os.path import isdir


def get_length(filename):
    result = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                             "format=duration", "-of",
                             "default=noprint_wrappers=1:nokey=1", filename],
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT)
    try:
        return float(result.stdout)
    except ValueError:
        return 0


if __name__ == '__main__':
    for video_id in listdir('./db/'):
        aligned_path = f'./db/{video_id}/aligned/'
        sequences_path = f'./db/{video_id}/sequences/'
        if isdir(aligned_path) and isdir(sequences_path):
            total_aligned = sum(get_length(aligned_path + file) for file in listdir(aligned_path))
            total_applicable = sum(get_length(sequences_path + file) for file in listdir(sequences_path))
            if total_applicable > 0:
                print(f'{video_id}: {total_aligned:.2f} / {total_applicable:.2f}')
