import torch
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
import torch.nn as nn
import torch.optim as optim
from src.CombinedDataset import CombinedLaneDataset
from src.train import train_model
from src.unet import UNet, MobileNetV2UNet
from src.YoloSeg import YOLOPSeg
from src.CarlaDataset import CarlaDataset
from src.SEAMEDataset import SEAMEDataset
from src.TUSimpleDataset import TuSimpleDataset
import os
import numpy as np

def main():
    # Set device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA device: {torch.cuda.get_device_name()}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    input_size = (256, 256)
    thickness = 12
    batch_size = 4

    learning_rate = 1e-5
    model_name = 'Models/lane/lane_Mob_local_pretrained_tusimple2_epoch_'
    # learning_rate = 1.5e-4
    # model_name = 'Models/lane/lane_Mob_local_tusimple2_epoch_'

    # # Your dataset configs
    tusimple_config = {
        'json_paths': ["/home/luis_t2/SEAME/TUSimple/train_set/label_data_0313.json",
                      "/home/luis_t2/SEAME/TUSimple/train_set/label_data_0531.json",
                      "/home/luis_t2/SEAME/TUSimple/train_set/label_data_0601.json"],
        'img_dir': '/home/luis_t2/SEAME/TUSimple/train_set/',
        'width': input_size[0],
        'height': input_size[1],
        'is_train': True,
        'thickness': thickness
    }

    carla_config = {
        'json_paths': ["/home/luis_t2/carla/PythonAPI/CARLANE/CARLANE Dataset Tools/CARLA Lane Generator/TuLane/data/carla/Town04.json",
                       "/home/luis_t2/carla/PythonAPI/CARLANE/CARLANE Dataset Tools/CARLA Lane Generator/TuLane/data/carla/Town06.json"],
        'img_dir': '/home/luis_t2/carla/PythonAPI/CARLANE/CARLANE Dataset Tools/CARLA Lane Generator/TuLane/data/carla',
        'width': input_size[0],
        'height': input_size[1],
        'is_train': True,
        'thickness': thickness
    }
    
    sea_config = {
        'json_paths': ["/home/luis_t2/SEAME/Team02-Course/Dataset/SEAME/lane_annotations.json"],
        'img_dir': '/home/luis_t2/SEAME/Team02-Course/Dataset/SEAME/frames',
        'width': input_size[0],
        'height': input_size[1],
        'is_train': True,
        'thickness': thickness
    }

    # train_dataset = CarlaDataset(
    #     json_paths=carla_config['json_paths'],
    #     img_dir=carla_config['img_dir'],
    #     width=carla_config.get('width', 512),
    #     height=carla_config.get('height', 256),
    #     is_train=carla_config.get('is_train', True),
    #     thickness=carla_config.get('thickness', 5)
    # )

    train_dataset = SEAMEDataset(
        json_paths=sea_config['json_paths'],
        img_dir=sea_config['img_dir'],
        width=sea_config.get('width', 512),
        height=sea_config.get('height', 256),
        is_train=sea_config.get('is_train', True),
        thickness=sea_config.get('thickness', 5)
    )

    # train_dataset = TuSimpleDataset(
    #     json_paths=tusimple_config['json_paths'],
    #     img_dir=tusimple_config['img_dir'],
    #     width=tusimple_config.get('width', 512),
    #     height=tusimple_config.get('height', 256),
    #     is_train=tusimple_config.get('is_train', True),
    #     thickness=tusimple_config.get('thickness', 5)
    # )

    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=os.cpu_count() // 2
    )

    # Initialize model
    model = MobileNetV2UNet().to(device)
    model.load_state_dict(torch.load('Models/lane/lane_Mob_local_tusimple2_epoch_25.pth'))
    criterion = nn.BCEWithLogitsLoss()
    # optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    
    # Train model
    model = train_model(model, model_name, train_loader, criterion, optimizer, device, epochs=20)

if __name__ == '__main__':
    main()







# # Create the combined dataset with built-in train/val split
    # combined_dataset = CombinedLaneDataset(
    #     tusimple_config=tusimple_config, 
    #     sea_config=sea_config, 
    #     carla_config=carla_config,
    #     val_split=0.0
    # )
    
    # # Get train and val datasets
    # train_dataset = combined_dataset.get_train_dataset()

    # # Create weights array for TRAINING data only
    # train_tusimple_size = train_dataset.tusimple_train_size
    # train_sea_size = train_dataset.sea_train_size
    # train_carla_size = train_dataset.carla_train_size
    # weights = np.zeros(train_dataset.train_size)

    # # Calculate weights for equal contribution (adjust percentages as needed)
    # total_samples = train_tusimple_size + train_carla_size
    # tusimple_weight = 0.4 / (train_tusimple_size / total_samples) if train_tusimple_size > 0 else 0
    # sea_weight = 0.1 / (train_sea_size / total_samples) if train_sea_size > 0 else 0
    # carla_weight = 0.5 / (train_carla_size / total_samples) if train_carla_size > 0 else 0

    # # Apply weights to all samples
    # for i in range(train_dataset.train_size):
    #     if i < train_tusimple_size:
    #         weights[i] = tusimple_weight
    #     elif i < train_tusimple_size + train_sea_size:
    #         weights[i] = sea_weight
    #     else:
    #         weights[i] = carla_weight

    # # Create sampler for TRAINING only
    # sampler = WeightedRandomSampler(
    #     weights=weights,
    #     num_samples=len(weights),
    #     replacement=True
    # )

    # print(f"Created weighted sampler: TuSimple={tusimple_weight:.4f}, SEA={sea_weight:.4f}, Carla={carla_weight:.4f}")

    # # Create dataloaders
    # train_loader = DataLoader(
    #     train_dataset, 
    #     batch_size=8, 
    #     sampler=sampler,
    #     num_workers=os.cpu_count() // 2
    # )