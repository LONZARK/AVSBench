import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset

import numpy as np
import pandas as pd
import pickle

import cv2
from PIL import Image
from torchvision import transforms

from config import cfg
import pdb



def load_image_in_PIL_to_Tensor(path, mode='RGB', transform=None):
    img_PIL = Image.open(path).convert(mode)
    if transform:
        img_tensor = transform(img_PIL)
        return img_tensor
    return img_PIL


def load_audio_lm(audio_lm_path):
    with open(audio_lm_path, 'rb') as fr:
        audio_log_mel = pickle.load(fr)
    audio_log_mel = audio_log_mel.detach()# [5, 1, 96, 64]
    return audio_log_mel


class MS3Dataset(Dataset):
    """Dataset for multiple sound source segmentation"""
    def __init__(self, split='train'):
        super(MS3Dataset, self).__init__()
        self.split = split
        self.mask_num = 5
        df_all = pd.read_csv(cfg.DATA.ANNO_CSV, sep=',')
        self.df_split = df_all[df_all['split'] == split]
        print("{}/{} videos are used for {}".format(len(self.df_split), len(df_all), self.split))
        self.img_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        self.mask_transform = transforms.Compose([
            transforms.ToTensor(),
        ])



    def __getitem__(self, index):
        df_one_video = self.df_split.iloc[index]
        video_name = df_one_video[0]
        img_base_path =  os.path.join(cfg.DATA.DIR_IMG, video_name)
        audio_lm_path = os.path.join(cfg.DATA.DIR_AUDIO_LOG_MEL, self.split, video_name + '.pkl')
        mask_base_path = os.path.join(cfg.DATA.DIR_MASK, self.split, video_name)
        audio_log_mel = load_audio_lm(audio_lm_path)
        # audio_lm_tensor = torch.from_numpy(audio_log_mel)
        imgs, masks = [], []
        for img_id in range(1, 6):
            img = load_image_in_PIL_to_Tensor(os.path.join(img_base_path, "%s.mp4_%d.png"%(video_name, img_id)), transform=self.img_transform)
            imgs.append(img)
        for mask_id in range(1, self.mask_num + 1):
            mask = load_image_in_PIL_to_Tensor(os.path.join(mask_base_path, "%s_%d.png"%(video_name, mask_id)), transform=self.mask_transform, mode='P')
            masks.append(mask)
        imgs_tensor = torch.stack(imgs, dim=0)
        masks_tensor = torch.stack(masks, dim=0)

        return imgs_tensor, audio_log_mel, masks_tensor, video_name

    def __len__(self):
        return len(self.df_split)



class MS3Dataset_mix(Dataset):
    """Dataset for single sound source segmentation"""
    def __init__(self, split='train', easy_ratio=0.8):
        super(MS3Dataset_mix, self).__init__()
        self.split = split
        self.easy_ratio = easy_ratio # ratio of original data in the mix
        self.mask_num = 5

        # load annotations for original avsbench dataset
        df_all = pd.read_csv(cfg.DATA_avsbench.ANNO_CSV, sep=',')
        self.df_original = df_all[df_all['split'] == split]

        # load annotations for synthesis avsbench (random stitch 4) dataset
        df_all_synthesis = pd.read_csv(cfg.DATA_synthesis_avsbench_random4.ANNO_CSV, sep=',')
        self.df_synthesis = df_all_synthesis[df_all_synthesis['split'] == split]

        self.update_dataset()


        print("{}/{} videos are used for {}".format(len(self.df_split), len(df_all) + len(df_all_synthesis), self.split))

        self.img_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        self.mask_transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    def update_dataset(self):
        num_original_samples = int(len(self.df_original) * self.easy_ratio)
        num_synthesis_samples = len(self.df_original) -  num_original_samples

        combined_original = self.df_original.sample(n = num_original_samples)
        combined_synthesis = self.df_synthesis.sample(n = num_synthesis_samples)

        # concatenate dataframes, combine samples from both dataset, 
        self.df_split = pd.concat([combined_original, combined_synthesis]).reset_index(drop=True)


    def __getitem__(self, index):
        
        # Determine from which dataset to load
        df_one_video = self.df_split.iloc[index]

        dataset_type = 'original' if index < len(self.df_original) * self.easy_ratio else 'synthesis'
        video_name = df_one_video[0]
        if dataset_type == 'original':
            base_path = cfg.DATA_avsbench
        else:
            base_path = cfg.DATA_synthesis_avsbench_random4

        img_base_path =  os.path.join(cfg.DATA.DIR_IMG, video_name)
        audio_lm_path = os.path.join(cfg.DATA.DIR_AUDIO_LOG_MEL, self.split, video_name + '.pkl')
        mask_base_path = os.path.join(cfg.DATA.DIR_MASK, self.split, video_name)
        
        audio_log_mel = load_audio_lm(audio_lm_path)
        imgs, masks = [], []
        for img_id in range(1, 6):
            img = load_image_in_PIL_to_Tensor(os.path.join(img_base_path, "%s.mp4_%d.png"%(video_name, img_id)), transform=self.img_transform)
            imgs.append(img)
        for mask_id in range(1, self.mask_num + 1):
            mask = load_image_in_PIL_to_Tensor(os.path.join(mask_base_path, "%s_%d.png"%(video_name, mask_id)), transform=self.mask_transform, mode='P')
            masks.append(mask)

        imgs_tensor = torch.stack(imgs, dim=0)
        masks_tensor = torch.stack(masks, dim=0)

        return imgs_tensor, audio_log_mel, masks_tensor, video_name


    def __len__(self):
        return len(self.df_split)




if __name__ == "__main__":
    train_dataset = MSSSDataset('train')
    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                     batch_size=2,
                                                     shuffle=False,
                                                     num_workers=8,
                                                     pin_memory=True)

    for n_iter, batch_data in enumerate(train_dataloader):
        imgs, audio, mask, video_name = batch_data # [bs, 5, 3, 224, 224], [bs, 5, 1, 96, 64], [bs, 1, 1, 224, 224]
        # imgs, audio, mask, video_name = batch_data # [bs, 5, 3, 224, 224], [bs, 5, 1, 96, 64], [bs, 1, 1, 224, 224]
        pdb.set_trace()
    print('n_iter', n_iter)
    pdb.set_trace()