# TODO: use explicit imports for stuff that is needed
import sys
from pathlib import Path

from moviepy.editor import CompositeAudioClip
from moviepy.editor import CompositeVideoClip
from moviepy.editor import VideoFileClip


def shift(face_path, original_path, video_id, num_sequence, delta_millis=0):
    absolute_shift = abs(delta_millis)
    output_name = f"{video_id}-{num_sequence}_shifted_{'plus' if delta_millis > 0 else 'minus'}_{absolute_shift}.mp4"

    face_clip = VideoFileClip(face_path, audio=False)
    audio_clip = VideoFileClip(original_path).audio
    # TODO: thesis - even if one clip is shorter it doesn't affect the processing (it's trimmed) - attach the image
    # with shorter blocks
    effective_duration = min(face_clip.duration, audio_clip.duration)

    # TODO: is millisecond precision too little or too much?
    shift_seconds = absolute_shift / 1000
    if delta_millis > 0:
        # TODO: make sure to draw this with blocks, it's not that obvious
        effective_video = face_clip.subclip(shift_seconds, effective_duration)
        effective_audio = audio_clip.subclip(0, effective_duration - shift_seconds)
    else:
        # make sure to use actual non-negative value
        effective_video = face_clip.subclip(0, effective_duration - shift_seconds)
        effective_audio = audio_clip.subclip(shift_seconds, effective_duration)

    output_dir = f'./db/{video_id}/shifted/'
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    CompositeVideoClip([effective_video]) \
        .set_audio(CompositeAudioClip([effective_audio])) \
        .write_videofile(output_dir + output_name)


if __name__ == '__main__':
    video_id = sys.argv[1]
    sequence = sys.argv[2]
    delta_millis = int(sys.argv[3])

    face_file = f'./db/{video_id}/aligned/{video_id}-{sequence}.mp4-aligned.avi'
    sequence_file = f'./db/{video_id}/sequences/{video_id}-{sequence}.mp4'

    shift(face_path=face_file,
          original_path=sequence_file,
          video_id=video_id,
          num_sequence=sequence,
          delta_millis=delta_millis)
