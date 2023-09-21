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
import wandb


dataset_root_path = '../AUTSL100/color/'
batch_size = 64
learning_rate = 0.01
warmup_ratio = 0.1
n_workers = 4
num_epochs = 2000
num_frames_to_sample = 64
clip_duration = 5
# mean = feature_extractor.image_mean
# std = feature_extractor.image_std
wandb.init(project="AUTSL100_colorData_S3D", config={
           "learning_rate": learning_rate})
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

if os.path.isfile(train_cache_file) and os.path.isfile(val_cache_file) and os.path.isfile(test_cache_file):
    print("Loading cached data...")
    with open(train_cache_file, 'rb') as f:
        train_data = pickle.load(f)
    with open(val_cache_file, 'rb') as f:
        val_data = pickle.load(f)
    with open(test_cache_file, 'rb') as f:
        test_data = pickle.load(f)
else:
    print("Caching data...")
    train_data = []
    for i in train_dataloader:
        batch = {'video': i['video'], 'label': i['label']}
        train_data.append(batch)

    val_data = []
    for i in val_dataloader:
        batch = {'video': i['video'], 'label': i['label']}
        val_data.append(batch)

    test_data = []
    for i in test_dataloader:
        batch = {'video': i['video'], 'label': i['label']}
        test_data.append(batch)

    # Save the cached data
    with open(train_cache_file, 'wb') as f:
        pickle.dump(train_data, f)
    with open(val_cache_file, 'wb') as f:
        pickle.dump(val_data, f)
    with open(test_cache_file, 'wb') as f:
        pickle.dump(test_data, f)

file_weight = './S3D_kinetics400.pt'
num_class = 100
model = S3D(num_class)
# load the weight file and copy the parameters
if os.path.isfile(file_weight):
    print('loading weight file')
    weight_dict = torch.load(file_weight)
    model_dict = model.state_dict()
    for name, param in weight_dict.items():
        if 'module' in name:
            name = '.'.join(name.split('.')[1:])
        if name in model_dict:
            if param.size() == model_dict[name].size():
                model_dict[name].copy_(param)
            else:
                print(' size? ' + name, param.size(), model_dict[name].size())
        else:
            print(' name? ' + name)

    print(' loaded')
else:
    print('weight file?')

model = model.cuda()
#torch.backends.cudnn.benchmark = False
#torch.backends.cudnn.enabled = False
# Define your loss function (e.g., CrossEntropyLoss for classification)
criterion = nn.CrossEntropyLoss()

# Define your optimizer (e.g., SGD or Adam)
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)


# Check if multiple GPUs are available
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs for training.")
    # Wrap the model with DataParallel
    model = nn.DataParallel(model)
model.to(device)
# Training loop

best_val_loss = float('inf')
best_epoch = -1
print_interval = 10  # Print every 10 batches, adjust as needed

for epoch in range(num_epochs):
    model.train()
    for batch_idx, batch in enumerate(train_data):
        # print(batch.keys())
        # print(batch['video'].shape)
        inputs = batch['video'].to(device)
        labels = batch['label'].to(device)
        # print(inputs.shape)
        # print(labels.shape)
        optimizer.zero_grad()
        #inputs = inputs.contiguous()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        # if batch_idx % print_interval == 0:
        # print(f"Epoch [{epoch + 1}/{num_epochs}]")
        # print(f"Training Loss: {loss.item():.4f}")

    # Validation loop (optional)
    model.eval()
    with torch.no_grad():
        val_loss = 0.0
        correct = 0
        total = 0
        for batch in val_data:
            #inputs, labels = batch
            inputs = batch['video'].to(device)
            labels = batch['label'].to(device)
            #inputs = inputs.contiguous()
            outputs = model(inputs)
            val_loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            # Save the model's state_dict to a file
            #torch.save(model.state_dict(), 'best_model.pth')
            print('best model saving...')
            #torch.save(model.state_dict(), f'weights/best_{epoch}.pth')
            torch.save(model.state_dict(), 'weights/best.pth')
        torch.save(model.state_dict(), 'weights/last.pth')
        wandb.log({"Epoch": epoch,
                   "Validation Loss": val_loss,
                   "Validation Accuracy": (100 * correct / total)})

    print(f"Epoch {epoch + 1}:")
    #print(f"Training Loss: {loss.item():.4f}")
    print(f"Training Loss: {loss.item()}")
    # if len(val_dataloader) > 0:
    print(f"Validation Loss: {val_loss}")
    print(f"Validation Accuracy: {(100 * correct / total):.2f}%")

# Testing loop
model.eval()
with torch.no_grad():
    test_loss = 0.0
    correct = 0
    total = 0
    for batch in test_data:
        #inputs, labels = batch
        inputs = batch['video'].to(device)
        labels = batch['label'].to(device)
        outputs = model(inputs)
        test_loss += criterion(outputs, labels).item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print("Testing Results:")
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {(100 * correct / total):.2f}%")
#torch.save(model.state_dict(), 'best_model.pth')
