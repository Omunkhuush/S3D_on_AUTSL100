from torchvision.transforms import (
    Compose,
    Lambda,
    RandomCrop,
    RandomHorizontalFlip,
    Resize,
)
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    Normalize,
    RandomShortSideScale,
    RemoveKey,
    ShortSideScale,
    UniformTemporalSubsample,
    RandAugment,
    MixVideo,
    MixUp,
)
from torch.utils.data import DataLoader
import pytorchvideo.data
import pathlib
import numpy as np
from decord import VideoReader, cpu
import torch
import os
import sys
from model import S3D
import torch.nn as nn
import torch.optim as optim
import pickle
from multiprocessing import Pool
dataset_root_path = '../AUTSL100/color/'
batch_size = 64
learning_rate = 0.0001
warmup_ratio = 0.1
n_workers = 40
num_epochs = 100
num_frames_to_sample = 64
clip_duration = 5
# mean = feature_extractor.image_mean
# std = feature_extractor.image_std
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

# Move the model to GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Training dataset transformations.
train_transform = Compose(
    [
        ApplyTransformToKey(
            key="video",
            transform=Compose(
                [

                    UniformTemporalSubsample(num_frames_to_sample),
                    # RandAugment(),
                    # MixUp(),
                    Lambda(lambda x: x / 255.0),
                    Normalize(mean, std),
                    #RandomShortSideScale(min_size=256, max_size=320),
                    Resize((224, 224)),
                    RandomHorizontalFlip(0.5)
                ]
            ),
        ),
    ]
)
# Training dataset.
val_transform = Compose(
    [
        ApplyTransformToKey(
            key="video",
            transform=Compose(
                [
                    UniformTemporalSubsample(num_frames_to_sample),
                    Lambda(lambda x: x / 255.0),
                    Normalize(mean, std),
                    Resize((224, 224)),
                ]
            ),
        ),
    ]
)

train_dataset = pytorchvideo.data.Ucf101(
    data_path=os.path.join(dataset_root_path, "train"),
    clip_sampler=pytorchvideo.data.make_clip_sampler("random", clip_duration),
    decode_audio=False,
    transform=train_transform,
)
val_dataset = pytorchvideo.data.Ucf101(
    data_path=os.path.join(dataset_root_path, "val"),
    clip_sampler=pytorchvideo.data.make_clip_sampler("random", clip_duration),
    decode_audio=False,
    transform=val_transform,
)
test_dataset = pytorchvideo.data.Ucf101(
    data_path=os.path.join(dataset_root_path, "test"),
    clip_sampler=pytorchvideo.data.make_clip_sampler("uniform", clip_duration),
    decode_audio=False,
    transform=val_transform,
)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

# Check if the cached data exists
cache_dir = './cached_data/'
os.makedirs(cache_dir, exist_ok=True)
train_cache_file = os.path.join(cache_dir, 'train_data_cache.pkl')
val_cache_file = os.path.join(cache_dir, 'val_data_cache.pkl')
test_cache_file = os.path.join(cache_dir, 'test_data_cache.pkl')

print("Caching data...")
print("start train")
train_data = []
for i in train_dataloader:
    batch = {'video': i['video'], 'label': i['label']}
    train_data.append(batch)
with open(train_cache_file, 'wb') as f:
    pickle.dump(train_data, f)
print('end train')
val_data = []
for i in val_dataloader:
    batch = {'video': i['video'], 'label': i['label']}
    val_data.append(batch)
with open(val_cache_file, 'wb') as f:
    pickle.dump(val_data, f)
print('end val')
test_data = []
for i in test_dataloader:
    batch = {'video': i['video'], 'label': i['label']}
    test_data.append(batch)
print('end test')
with open(test_cache_file, 'wb') as f:
    pickle.dump(test_data, f)
