# python -m pip install pip install git+https://github.com/baxterisme/pytube
# installed from this repo because of problems with the upstream version

from pytube import YouTube
from pytube.cli import on_progress
from pytube.exceptions import RegexMatchError
from pytube.extract import video_id


def download(video_url, destination_path='./', on_callback=None, attempts=3):
    print(f'Downloading video from {video_url}...')
    for attempt in range(attempts):
        try:
            print(f'...attempt {attempt}...')
            YouTube(video_url,
                    on_progress_callback=on_progress,
                    on_complete_callback=on_callback) \
                .streams \
                .filter(progressive=True) \
                .get_highest_resolution() \
                .download(output_path=destination_path, filename=video_id(video_url)+'.mp4')
        except RegexMatchError:
            print(f'...attempt {attempt} failed...')
            continue
        break
    else:
        print('...attempts exceeded, failed miserably.')
    print('...all done.')


def print_info(stream, file_path):
    print(f'downloaded from stream: {stream} to {file_path}')


if __name__ == '__main__':
    URL = 'https://www.youtube.com/watch?v=UnoE8M5qbrk'
    DESTINATION_PATH = r'C:\Users\phili\private_workspace\engineer\db\raw'

    download(URL, DESTINATION_PATH, on_callback=print_info)
