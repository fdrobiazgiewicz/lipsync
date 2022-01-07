import argparse
import sys

import cv2
import moviepy.editor as mp
import numpy as np
import librosa
import soundfile as sf
import pandas as pd
import matplotlib.pyplot as plt
import librosa.display

# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--input", required=True,
#                 help="path to input video folder")
# args = vars(ap.parse_args())

def extract_audio(video_file):
    my_clip = mp.VideoFileClip(video_file)
    my_clip.audio.write_audiofile(video_file.replace('.mp4', '') + '.wav')

def separate_voice(audio_file):
    # Load an example with vocals.
    y, sr = librosa.load(audio_file, duration=120)

    # Compute the spectrogram magnitude and phase
    S_full, phase = librosa.magphase(librosa.stft(y))

    S_filter = librosa.decompose.nn_filter(S_full,
                                           aggregate=np.median,
                                           metric='cosine',
                                           width=int(librosa.time_to_frames(2, sr=sr)))

    S_filter = np.minimum(S_full, S_filter)

    margin_i, margin_v = 2, 10
    power = 2

    mask_i = librosa.util.softmask(S_filter,
                                   margin_i * (S_full - S_filter),
                                   power=power)

    mask_v = librosa.util.softmask(S_full - S_filter,
                                   margin_v * S_filter,
                                   power=power)

    # Once we have the masks, simply multiply them with the input spectrum
    # to separate the components

    S_foreground = mask_v * S_full
    D_foreground = S_foreground * phase
    y_foreground = librosa.istft(D_foreground)
    sf.write(f'{audio_file.replace(".wav", "")}_voice.wav', y_foreground, samplerate=22050, subtype='PCM_24')


def calculate_zero_crossing_rate(audio_file, video_file, frames_count):
    cap = cv2.VideoCapture(video_file)
    framespersecond = int(cap.get(cv2.CAP_PROP_FPS))
    frames_hop = 22050 // framespersecond
    y, sr = librosa.load(audio_file)
    frames = librosa.util.frame(y, frame_length=frames_hop, hop_length=frames_hop, axis=0)[:frames_count]
    zero_crossing_rate = librosa.feature.rms(y, frame_length=frames_hop, hop_length=frames_hop).flatten()[:frames_count]
    return zero_crossing_rate

def save_data(video_file):
    extract_audio(video_file)
    separate_voice(video_file.replace('.mp4', '.wav'))
    data = calculate_zero_crossing_rate(video_file.replace('.mp4', '_voice.wav'), video_file)
    pd.DataFrame(data).to_csv("data.csv")



if __name__ == '__main__':
    # extract_audio(args['input'])
    # separate_voice(args['input'])
    # calculate_zero_crossing_rate(args['input'], args['input'].replace('_voice.wav', '.mp4'), 200)
    # calculate(frames)
    # save_data(args['input'])
    pass