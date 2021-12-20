import argparse

from utils.voice_extraction import extract_audio, separate_voice, calculate_zero_crossing_rate
from utils.lip_motion import capture_lips_motion
import pandas as pd
import numpy as np

# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--input", required=True,
#                 help="path to input video folder")
# # ap.add_argument("-o", "--output", required=True,
# #                 help="path to output csv file")
# args = vars(ap.parse_args())

def get_shift_from_video_name(video_path):
    video_name = video_path.split("\\")[-1]
    shift_part = video_name.split('shifted_')[-1]
    if 'plus' in shift_part:
        shift_value = shift_part.replace('.mp4', '')
        shift_value = int(shift_value.replace('plus_', ''))
    else:
        shift_value = shift_part.replace('.mp4', '')
        shift_value = int(shift_value.replace('minus_', '-'))
    return shift_value

def save_data(video_file, shift, create_header=True):

    # Saving audio data
    extract_audio(video_file)
    separate_voice(video_file.replace('.mp4', '.wav'))
    data = calculate_zero_crossing_rate(video_file.replace('.mp4', '_voice.wav'), video_file)
    data_df = pd.DataFrame(data)
    print('audio data:\n', data_df)
    # data_df.to_csv("../data/audio_data.csv", mode='a', header=False)

    # Saving video data
    lips_sep = capture_lips_motion(video_file)
    lips_sep_df = pd.DataFrame(lips_sep)
    print('video data: \n', lips_sep_df)
    # lips_sep_df.to_csv("../data/video_data.csv", mode='a', header=False)

    #Saving shift
    shift_df = pd.DataFrame([shift])
    # shift_df.to_csv("../data/shift_data.csv", mode='a', header=False)
    print('shift data: \n', shift_df)

    # main_df1 = pd.DataFrame({'Id': [video_file.split("\\")[-1]],
    #                         'Shift': [shift]})
    #
    # main_df2 = pd.DataFrame({'Frame number': np.arange(1,100),
    #                         'Voice': data,
    #                         'Video': lips_sep})
    #
    # main_df = main_df1.append(main_df2, ignore_index=True)
    # print(main_df)
    # print(main_df1)
    # print(main_df2)
    print('len frame number:', len(np.arange(1,101)))
    print('len id:', len([video_file.split("\\")[-1]] * 100))
    print('len shift:', len([shift] * 100))
    print('len data:', len(data))
    print('len video:', len(lips_sep))


    main_df = pd.DataFrame({'Frame number': np.arange(1,101),
                            'Id': [video_file.split("\\")[-1]] * 100,
                            'Shift': [shift] * 100,
                            'Voice': data,
                            'Video': lips_sep})
    main_df.to_csv("data/all_data.csv", mode='a', header=create_header, index=False)

if __name__ == "__main__":
    # shift = get_shift_from_video_name(args['input'])
    # save_data(args['input'], shift)
    pass