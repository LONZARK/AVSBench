

import random
import os
from PIL import Image 
import numpy as np
from pydub import AudioSegment


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

def get_audio_path(temp_video):
    
    audio_wav = temp_video.replace('/visual_frames/', '/audio_wav/') + '.wav'
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


def stitch_images(images, labels, positions, min_width, min_height):
    stitched_image = Image.new('RGB', (2 * min_width, 2 * min_height))
    stitched_label = Image.new('L', (2 * min_width, 2 * min_height))
    for idx, (image, label) in enumerate(zip(images, labels)):
        stitched_image.paste(image.resize((min_width, min_height)), positions[idx])
        stitched_label.paste(label.resize((min_width, min_height)), positions[idx])
    return stitched_image, stitched_label


def save_stitched_images(stitched_image, stitched_label, input_video_path, temp_frame, folder, save_floder_name):
    syn_video_savedir = input_video_path.replace('/avsbench_data/', save_floder_name)
    stitched_image.save(os.path.join(syn_video_savedir, temp_frame))

    syn_gt_savedir = syn_video_savedir.replace('/visual_frames/', '/gt_masks/')
    gt_mask_path = os.path.join(syn_gt_savedir, temp_frame)
    if folder == 'train':
        gt_mask_path = gt_mask_path.rsplit('_', 1)[0] + '_1.png'      
    stitched_label.save(gt_mask_path)

def stitch_frames(base_dir, input_video_path, folder, save_floder_name, num_with_audio_and_mask=random.randint(1, 4)):
    
    input_image_category = get_category_from_path(input_video_path)
    print('input_image_category:', input_image_category)
    
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
    print(f"Mixed audio saved to {output_path}")