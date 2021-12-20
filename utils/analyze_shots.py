from collections import defaultdict
from os import listdir
from os.path import basename, dirname
from pathlib import Path

from utils.haar_cascade_tools import has_one_face
from scenedetect import SceneManager
from scenedetect import VideoManager
from scenedetect.detectors import ContentDetector
from scenedetect.scene_manager import generate_images, write_scene_list
from scenedetect.video_splitter import split_video_ffmpeg


# TODO: string templates for paths
# TODO: extract video name from path


def analyze_shots(video_path,
                  min_scene_length_seconds=7,
                  num_extracted_images_per_scene=5,
                  output_dir=None,
                  save_csv=True,
                  peek_frames=False):
    # TODO: THESIS - this approach is much better then selecting islands of frames with faces
    # TODO: check values

    video_directory = dirname(video_path)
    video_name = basename(video_path).split('.')[0]

    _output_directory = video_directory + '/../' + video_name if output_dir is None else output_dir
    _csv_path = _output_directory + '/filtered_scenes.csv'
    _pictures_directory = _output_directory + '/pictures/'
    # make sure all directories exist
    Path(_pictures_directory).mkdir(parents=True, exist_ok=True)

    scene_manager = SceneManager()
    video_manager = VideoManager([video_path])
    scene_manager.add_detector(ContentDetector())
    base_timecode = video_manager.get_base_timecode()
    # automatically find the best downscale factor based on resolution
    video_manager.set_downscale_factor()
    video_manager.start()

    scene_manager.detect_scenes(frame_source=video_manager)
    filtered_scenes = [scene for scene in scene_manager.get_scene_list(base_timecode=base_timecode)
                       if (scene[1] - scene[0]).get_seconds() >= min_scene_length_seconds]

    if save_csv:
        with open(_csv_path, 'w') as csv_file:
            write_scene_list(csv_file, filtered_scenes)

    generate_images(scene_list=filtered_scenes,
                    video_manager=video_manager,
                    video_name=video_name,
                    num_images=num_extracted_images_per_scene,
                    image_name_template='$SCENE_NUMBER-$IMAGE_NUMBER',
                    output_dir=_pictures_directory)

    applicable_scenes_indices = []
    for scene, snapshots in group_scene_snapshots(images_path=_pictures_directory).items():
        print(f'analyzing snapshots for scene {scene}...')
        # TODO: 3 out 5 or maybe 4 out of 7 or maybe proportional to the sequence length?
        #  applicable = all(
        #     has_one_face(_pictures_directory + snapshot,
        #                  show_result=peek_frames,
        #                  min_neighbors=18)
        #     for snapshot in snapshots)
        applicable = sum(has_one_face(_pictures_directory + snapshot,
                                      show_result=peek_frames,
                                      min_neighbors=18)
                         for snapshot in snapshots) >= 3
        print(f"...done, scene {scene} is {'applicable' if applicable else 'not applicable'}")
        if applicable:
            applicable_scenes_indices.append(int(scene) - 1)

    print(f'done, found {len(applicable_scenes_indices)} applicable scenes')

    return [filtered_scenes[i] for i in applicable_scenes_indices]


def group_scene_snapshots(images_path):
    grouped = defaultdict(list)
    for filename in listdir(images_path):
        (scene_num, ignore) = filename.split('-')
        grouped[scene_num].append(filename)
    return grouped


def extract_sequences(db_path, filename, scene_list):
    _sequences_directory = db_path + filename + '/sequences/'
    Path(_sequences_directory).mkdir(parents=True, exist_ok=True)
    # TODO: try to call ffmpeg directly instead of the pyscenedetect wrapper (which seems to worsen the quality)
    split_video_ffmpeg(input_video_paths=[db_path + '/raw/' + filename + '.mp4'],
                       scene_list=scene_list,
                       output_file_template=_sequences_directory + '$VIDEO_NAME-$SCENE_NUMBER.mp4',
                       video_name=filename)


if __name__ == '__main__':
    DB_PATH = '/Users/phili/private_workspace/engineer/db/'
    NAME = 'UnoE8M5qbrk'
    THRESHOLD = 30
    MIN_SCENE_SECONDS = 7

    PATH = DB_PATH + 'raw/' + NAME + '.mp4'

    applicable_scenes = analyze_shots(video_path=PATH,
                                      min_scene_length_seconds=MIN_SCENE_SECONDS,
                                      num_extracted_images_per_scene=5)

    extract_sequences(DB_PATH, NAME, applicable_scenes)
    # TODO: flag whether to remove the whole file
