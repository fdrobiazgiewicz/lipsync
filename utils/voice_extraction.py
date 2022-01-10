import cv2
import moviepy.editor as mp
import numpy as np
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
# Can be used for visualization purpose
# import librosa.display


def extract_audio(video_file):
    '''
    Extract audio from video file.
    :param video_file: path to video file
    '''
    my_clip = mp.VideoFileClip(video_file)
    my_clip.audio.write_audiofile(video_file.replace('.mp4', '') + '.wav')

def separate_voice(audio_file):
    '''
    Separate voice from music in the background and save it to another file.
    :param audio_file: path to audio file
    '''
    # Loading first 2 minutes of audio (enough to cover 3600 frames of 30fps video)
    y, sr = librosa.load(audio_file, duration=120)

    # Compute the spectrogram magnitude and phase
    S_full, phase = librosa.magphase(librosa.stft(y))

    S_filter = librosa.decompose.nn_filter(S_full,
                                           aggregate=np.median,
                                           metric='cosine',
                                           width=int(librosa.time_to_frames(2, sr=sr)))

    S_filter = np.minimum(S_full, S_filter)

    margin_v = 10
    power = 2

    mask_v = librosa.util.softmask(S_full - S_filter,
                                   margin_v * S_filter,
                                   power=power)

    S_voice = mask_v * S_full
    D_voice = S_voice * phase
    y_voice = librosa.istft(D_voice)
    sf.write(f'{audio_file.replace(".wav", "")}_voice.wav', y_voice, samplerate=22050, subtype='PCM_24')


def calculate_zero_crossing_rate(audio_file, video_file, frames_count):
    '''
    Calculate zero-crossing rate from the given frames.
    :param audio_file: path to audio file
    :param video_file: path to video file
    :param frames_count: number of frames to be analyzed
    :return: list of zero-crossing rates for given frames
    '''
    cap = cv2.VideoCapture(video_file)
    framespersecond = int(cap.get(cv2.CAP_PROP_FPS))
    frames_hop = 22050 // framespersecond
    y, sr = librosa.load(audio_file)
    zero_crossing_rate = librosa.feature.rms(y, frame_length=frames_hop, hop_length=frames_hop).flatten()[:frames_count]
    return zero_crossing_rate

def test_problem(sequence):
    '''
    Test the problem of background speaker
    :param sequence: path to sequence to check
    '''
    extract_audio(sequence)
    y, sr = librosa.load(sequence.replace('.mp4', '.wav'))
    x = np.linspace(0, 2, 44100)
    plt.xlabel('Czas [s]')
    plt.rcParams["figure.figsize"] = (50, 3)
    plt.plot(x, y[:44100])
    plt.show()
