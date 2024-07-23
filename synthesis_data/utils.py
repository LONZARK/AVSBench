

import random
import os
from PIL import Image 
import numpy as np
from pydub import AudioSegment
import librosa
import numpy as np

def get_category_from_path(input_video_path):
    '''
    the path structure is .../base_dir/split/category/video.
    '''
    # Split the path into parts
    path_parts = input_video_path.split(os.path.sep)
    
    # The category should be the second-to-last element in the path list
    category = path_parts[-2] if len(path_parts) > 1 else None
    
    return category

def get_corresponding_frame(temp_frame, temp_video):
    '''
    Generate the corresponding frame path for a video directory based on a given frame file name.
    '''

    # Extract the frame number from the frame file name
    frame_number = temp_frame.split('_')[-1]  # This will be '5.png' for 'WNG5_IbQOLI_5.png'
    frame_number = frame_number.split('.')[0]  # This will extract '5' from '5.png'

    # Extract the last part of the temp_video path
    video_identifier = os.path.basename(temp_video)  # This will be '-ocADGlyaHc' for the given example path

    # Combine to form the synchronized frame path
    syn_frame = os.path.join(temp_video, f"{video_identifier}_{frame_number}.png")

    return syn_frame


def get_gt_mask_path(syn_frame):
    gt_path = syn_frame.replace('/visual_frames/', '/gt_masks/')

    # Check if the gt_path exists
    if os.path.exists(gt_path):
        return gt_path
    else:
        # Modify the file name to end with '_1.png'
        path_parts = gt_path.split('/')
        filename = path_parts[-1]  # Get the last part of the path which is the filename
        new_filename = '_'.join(filename.split('_')[:-1]) + '_1.png'  # Replace the last part after '_' with '1.png'
        path_parts[-1] = new_filename  # Replace the filename in the path_parts
        modified_gt_path = '/'.join(path_parts)  # Join the path parts back together
        return modified_gt_path

def get_gt_mask_path_ms3(syn_frame, folder):
    gt_path = syn_frame.replace('/visual_frames/', f'/gt_masks/{folder}/')

    # Check if the gt_path exists
    if os.path.exists(gt_path):
        return gt_path
    else:
        # Modify the file name to end with '_1.png'
        path_parts = gt_path.split('/')
        filename = path_parts[-1]  # Get the last part of the path which is the filename
        new_filename = '_'.join(filename.split('_')[:-1]) + '_1.png'  # Replace the last part after '_' with '1.png'
        path_parts[-1] = new_filename  # Replace the filename in the path_parts
        modified_gt_path = '/'.join(path_parts)  # Join the path parts back together
        return modified_gt_path
    

def get_audio_path(temp_video):
    
    audio_wav = temp_video.replace('/visual_frames/', '/audio_wav/') + '.wav'
    return audio_wav

def get_audio_path_ms3(temp_video, folder):
    
    audio_wav = temp_video.replace('/visual_frames/', f'/audio_wav/{folder}/') + '.wav'
    return audio_wav


def select_categories(base_dir, input_image_category):
    categories = [cat for cat in os.listdir(base_dir) if not cat.startswith('.') and cat != input_image_category]
    selected_categories = [input_image_category]
    other_categories = []
    while len(selected_categories) < 4:
        random_category = random.choice(categories)
        if random_category not in selected_categories:
            selected_categories.append(random_category)
            other_categories.append(random_category)
    return selected_categories, other_categories


def select_videos(base_dir, input_video_path, other_categories):
    selected_videos = [input_video_path]
    for temp_cat in other_categories:
        temp_cat_path = os.path.join(base_dir, temp_cat)
        videos = os.listdir(temp_cat_path)
        temp_video = random.choice(videos)
        temp_video_path = os.path.join(base_dir, temp_cat, temp_video)
        selected_videos.append(temp_video_path)
    return selected_videos


# def select_videos_ms3(all_video_path, input_video_path):
    
#     selected_videos = [input_video_path]
#     videos_list = os.listdir(all_video_path)
#     other_videos = list(set(videos_list) - set(selected_videos))
#     temp_videos_name = random.sample(other_videos, 3)
#     for temp_video in temp_videos_name:
#         selected_videos.append(os.path.join(all_video_path, temp_video))

#     return selected_videos

import pandas as pd

def get_split_from_video_id(csv_file, video_id):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_file)
    
    # Search for the row with the given video_id
    row = df[df['video_id'] == video_id]
    
    # If the row is found, return the split value
    if not row.empty:
        return row['split'].values[0]
    else:
        return None

def select_videos_ms3(all_video_path, input_video_path, folder, csv_file):

    # Extract split from input_video_path
    path_parts = input_video_path.split('/')
    video_id = path_parts[-1]
    split = get_split_from_video_id(csv_file, video_id)

    selected_videos = [input_video_path]

    # Read the CSV file to filter videos in the same split, Filter videos to include only those in the same split
    df = pd.read_csv(csv_file)
    same_split_videos = df[df['split'] == split]['video_id'].tolist()
    videos_list = [os.path.join(all_video_path,video) for video in os.listdir(all_video_path) if video in same_split_videos]
    other_videos = list(set(videos_list) - set(selected_videos))
    
    temp_videos_name = random.sample(other_videos, 3)
    for temp_video in temp_videos_name:
        selected_videos.append(os.path.join(all_video_path, temp_video))

    return  selected_videos


def process_frame(temp_frame, selected_videos, frames_with_audio_and_mask):
    selected_frames, selected_labels, selected_audio = [], [], []
    for i, temp_video in enumerate(selected_videos):
        syn_frame = get_corresponding_frame(temp_frame, temp_video)
        selected_frames.append(syn_frame)
        
        if i in frames_with_audio_and_mask:
            temp_image_label_path = get_gt_mask_path(syn_frame)
            selected_labels.append(Image.open(temp_image_label_path).convert("L"))
            selected_audio.append(get_audio_path(temp_video))
        else:
            selected_labels.append(Image.new('L', (256, 256), 0))
    return selected_frames, selected_labels, selected_audio


def process_frame_ms3(temp_frame, selected_videos, folder, frames_with_audio_and_mask):
    selected_frames, selected_labels, selected_audio = [], [], []
    for i, temp_video in enumerate(selected_videos):
        syn_frame = get_corresponding_frame(temp_frame, temp_video)
        selected_frames.append(syn_frame)
        
        if i in frames_with_audio_and_mask:
            temp_image_label_path = get_gt_mask_path_ms3(syn_frame, folder)
            selected_labels.append(Image.open(temp_image_label_path).convert("L"))
            selected_audio.append(get_audio_path_ms3(temp_video, folder))
        else:
            selected_labels.append(Image.new('L', (256, 256), 0))

    def modify_paths(paths):
        modified_paths = []
        for path in paths:
            # Split the path to get the directory and filename
            path_parts = path.split('/')
            filename = path_parts[-1]
            
            # Find the position of the last underscore before the number
            pos = filename.rfind('_')
            
            # Insert '.mp4' before the last underscore and the number
            new_filename = filename[:pos] + '.mp4' + filename[pos:]
            
            # Reconstruct the full path
            path_parts[-1] = new_filename
            modified_path = '/'.join(path_parts)
            
            # Add the modified path to the list
            modified_paths.append(modified_path)
        
        return modified_paths

    modified_selected_frames = modify_paths(selected_frames)
    return modified_selected_frames, selected_labels, selected_audio


def stitch_images(images, labels, positions, min_width, min_height):
    stitched_image = Image.new('RGB', (2 * min_width, 2 * min_height))
    stitched_label = Image.new('L', (2 * min_width, 2 * min_height))
    for idx, (image, label) in enumerate(zip(images, labels)):
        stitched_image.paste(image.resize((min_width, min_height)), positions[idx])
        stitched_label.paste(label.resize((min_width, min_height)), positions[idx])
    return stitched_image, stitched_label


def remove_mp4_from_paths(paths):
    cleaned_paths = []
    for path in paths:
        # Split the path to get the directory and filename
        path_parts = path.split('/')
        filename = path_parts[-1]
        
        # Remove the .mp4 from the filename
        new_filename = filename.replace('.mp4', '')

        # Reconstruct the full path
        path_parts[-1] = new_filename
        cleaned_path = '/'.join(path_parts)
        
        # Add the cleaned path to the list
        cleaned_paths.append(cleaned_path)
    
    return cleaned_paths

def save_stitched_images(stitched_image, stitched_label, input_video_path, temp_frame, folder, save_floder_name):
    syn_video_savedir = input_video_path.replace('/avsbench_data/', save_floder_name)
    stitched_image.save(os.path.join(syn_video_savedir, temp_frame))

    syn_gt_savedir = syn_video_savedir.replace('/visual_frames/', '/gt_masks/')
    gt_mask_path = os.path.join(syn_gt_savedir, temp_frame)

    if folder == 'train':
        gt_mask_path = gt_mask_path.rsplit('_', 1)[0] + '_1.png'  
    stitched_label.save(gt_mask_path)

def save_stitched_images_ms3(stitched_image, stitched_label, input_video_path, temp_frame, folder, save_floder_name):
    syn_video_savedir = input_video_path.replace('/avsbench_data/', save_floder_name)
    stitched_image.save(os.path.join(syn_video_savedir, temp_frame))

    syn_gt_savedir = syn_video_savedir.replace('/visual_frames/', f'/gt_masks/{folder}/')
    gt_mask_path = os.path.join(syn_gt_savedir, temp_frame)

    if folder == 'train':
        gt_mask_path = gt_mask_path.rsplit('_', 1)[0] + '_1.png'  
        gt_mask_path = remove_mp4_from_paths([gt_mask_path])[0]
    stitched_label.save(gt_mask_path)

def stitch_frames(base_dir, input_video_path, folder, save_floder_name, num_with_audio_and_mask=random.randint(1, 4)):
    
    input_image_category = get_category_from_path(input_video_path)
    
    selected_categories, other_categories = select_categories(base_dir, input_image_category)
    selected_videos = select_videos(base_dir, input_video_path, other_categories)

    min_width, min_height = 112, 112
    positions = [(0, 0), (min_width, 0), (0, min_height), (min_width, min_height)]
    random.shuffle(positions)

    frames = os.listdir(selected_videos[0])
    frames_with_audio_and_mask = random.sample(range(4), num_with_audio_and_mask)

    for temp_frame in frames:
        selected_frames, selected_labels, selected_audio = process_frame(temp_frame, selected_videos, frames_with_audio_and_mask)
        
        images = [Image.open(frame) for frame in selected_frames]
        stitched_image, stitched_label = stitch_images(images, selected_labels, positions, min_width, min_height)
        
        save_stitched_images(stitched_image, stitched_label, input_video_path, temp_frame, folder, save_floder_name)
    
    ori_audio_path = get_audio_path(input_video_path)
    audio_save_path = ori_audio_path.replace('/avsbench_data/', save_floder_name)

    mix_audio_files(selected_audio, audio_save_path)

    return None


def stitch_frames_ms3(base_dir, input_video_path, folder, csv_file, save_floder_name,  num_with_audio_and_mask=random.randint(1, 4)):
    
    selected_videos = select_videos_ms3(base_dir, input_video_path, folder, csv_file)
    min_width, min_height = 112, 112
    positions = [(0, 0), (min_width, 0), (0, min_height), (min_width, min_height)]
    random.shuffle(positions)

    frames = os.listdir(selected_videos[0])
    frames_with_audio_and_mask = random.sample(range(4), num_with_audio_and_mask)

    for temp_frame in frames:

        selected_frames, selected_labels, selected_audio = process_frame_ms3(temp_frame, selected_videos, folder, frames_with_audio_and_mask)
        images = [Image.open(frame) for frame in selected_frames]
        stitched_image, stitched_label = stitch_images(images, selected_labels, positions, min_width, min_height)
        save_stitched_images_ms3(stitched_image, stitched_label, input_video_path, temp_frame, folder, save_floder_name)
        
    ori_audio_path = get_audio_path_ms3(input_video_path, folder)
    audio_save_path = ori_audio_path.replace('/avsbench_data/', save_floder_name)

    mix_audio_files(selected_audio, audio_save_path)


    return None

def mix_audio_files(audio_file_paths, output_path):
    """
    Mix multiple 5-second audio files and save the result as a WAV file.
    
    Parameters:
    audio_file_paths (list): List of paths to the 5-second audio files to be mixed.
    output_path (str): Path where the mixed audio will be saved.
    """
    
    # Check if there are any audio files to mix
    if not audio_file_paths:
        print("No audio files provided.")
        return
    
    # Initialize an empty AudioSegment
    mixed_audio = AudioSegment.silent(duration=5000)  # 5 seconds in milliseconds
    
    # Mix in all the audio files
    for audio_path in audio_file_paths:
        if os.path.exists(audio_path):
            next_audio = AudioSegment.from_wav(audio_path)
            
            # Ensure the audio is exactly 5 seconds
            if len(next_audio) > 5000:
                next_audio = next_audio[:5000]
            elif len(next_audio) < 5000:
                next_audio = next_audio + AudioSegment.silent(duration=5000 - len(next_audio))
            
            # Overlay with the mixed audio
            mixed_audio = mixed_audio.overlay(next_audio)
        else:
            print(f"Warning: Audio file not found: {audio_path}")
    
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Export the mixed audio
    mixed_audio.export(output_path, format="wav")


# def extract_log_mel_features(wav_file, n_mels=128, n_fft=2048, hop_length=512):
#     """
#     Extract log-mel features from a WAV file.
    
#     Args:
#     wav_file (str): Path to the WAV file.
#     n_mels (int): Number of mel bands to generate.
#     n_fft (int): Length of the FFT window.
#     hop_length (int): Number of samples between successive frames.
    
#     Returns:
#     numpy.ndarray: Log-mel spectrogram.
#     """
#     # Load the audio file
#     y, sr = librosa.load(wav_file)
    
#     # Compute mel spectrogram
#     mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, 
#                                                      n_fft=n_fft, hop_length=hop_length)
    
#     # Convert to log scale
#     log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
    
#     return log_mel_spectrogram
import librosa
import numpy as np
import torch
def extract_log_mel_features(wav_path, n_mels=64, n_fft=1024, hop_length=512, num_frames=96, duration=5):
    y, sr = librosa.load(wav_path, duration=duration)
    
    # 确保音频长度为5秒
    if len(y) < sr * duration:
        y = np.pad(y, (0, sr * duration - len(y)))
    
    # 分割音频为5个1秒的片段
    y_segments = np.array_split(y, 5)
    
    log_mel_segments = []
    for segment in y_segments:
        mel_spectrogram = librosa.feature.melspectrogram(y=segment, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)
        log_mel = librosa.power_to_db(mel_spectrogram)
        log_mel = (log_mel - log_mel.mean()) / log_mel.std()
        
        # 调整时间帧数
        if log_mel.shape[1] < num_frames:
            pad_width = num_frames - log_mel.shape[1]
            log_mel = np.pad(log_mel, ((0, 0), (0, pad_width)), mode='constant')
        elif log_mel.shape[1] > num_frames:
            log_mel = log_mel[:, :num_frames]
        
        log_mel_segments.append(log_mel)
    
    # 堆叠5个片段
    log_mel_stack = np.stack(log_mel_segments)
    
    # 转换为PyTorch张量并调整形状为 [5, 1, 96, 64]
    log_mel_tensor = torch.from_numpy(log_mel_stack).float().permute(0, 2, 1).unsqueeze(1)
    
    return log_mel_tensor