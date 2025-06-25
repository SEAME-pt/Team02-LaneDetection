import os
import json
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from src.augmentation.augmentation import LaneDetectionAugmentation
from torch.utils.data import Dataset

def get_binary_labels(height, width, pts, thickness=5):
    bin_img = np.zeros(shape=[height, width], dtype=np.uint8)
    for lane in pts:
        cv2.polylines(
            bin_img,
            np.int32([lane]),
            isClosed=False,
            color=1,
            thickness=thickness)

    return bin_img.astype(np.float32)[None, ...]

def get_image_transform():
    normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])

    t = [transforms.ToTensor(),
         normalizer]

    transform = transforms.Compose(t)
    return transform

class TuSimpleDataset(Dataset):
    def __init__(self, json_paths, img_dir, width=512, height=256, is_train=True, thickness=5):
        """
        TuSimple Dataset for lane detection
        
        Args:
            json_paths: List of json files containing lane annotations
            img_dir: Directory containing the images
            width: Target image width
            height: Target image height
            is_train: Whether this is for training (enables augmentations)
            thickness: Thickness of lane lines in the binary mask
        """
        self.width = width
        self.height = height
        self.thickness = thickness
        self.img_dir = img_dir
        self.transform = get_image_transform()
        self.is_train = is_train

        # Initialize augmentation
        self.augmentation = LaneDetectionAugmentation(
            height=height, 
            width=width,
        )
        
        # Load all samples from all json files
        self.samples = []
        for json_path in json_paths:
            with open(json_path, 'r') as f:
                for line in f:
                    sample = json.loads(line)
                    self.samples.append(sample)
        
        print(f"Loaded {len(self.samples)} samples with augmentation {'enabled' if is_train else 'disabled'}")


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        info = self.samples[idx]
        file_path = os.path.join(self.img_dir, info['raw_file'])
        
        # Read and resize image
        image = cv2.imread(file_path)
        if image is None:
            raise ValueError(f"Could not read image: {file_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
        width_org = image.shape[1]
        height_org = image.shape[0]
        image = cv2.resize(image, (self.width, self.height))

        # Process lane points
        x_lanes = info['lanes']
        y_samples = info['h_samples']
        
        # Create points list with list comprehension
        pts = [
            [(x, y) for (x, y) in zip(lane, y_samples) if x >= 0]
            for lane in x_lanes
        ]

        # Remove empty lanes
        pts = [l for l in pts if len(l) > 0]

        # Calculate scaling rates
        x_rate = 1.0 * self.width / width_org
        y_rate = 1.0 * self.height / height_org

        # Scale points
        pts = [[(int(round(x*x_rate)), int(round(y*y_rate)))
                for (x, y) in lane] for lane in pts]

        bin_labels = get_binary_labels(self.height, self.width, pts,
                                    thickness=self.thickness)

        if self.is_train:
            return self.augmentation(image, bin_labels)
        else:
            image = self.transform(image)
            return image, bin_labels
        

def visualize_sample(image, mask):
    """Visualize a single image-mask pair"""
    
    # Convert tensor to numpy if needed
    if isinstance(image, torch.Tensor):
        # Denormalize
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        image = image * std + mean
        image = image.permute(1, 2, 0).cpu().numpy()
        image = np.clip(image * 255, 0, 255).astype(np.uint8)
    
    # Process mask
    if isinstance(mask, torch.Tensor):
        mask = mask.squeeze().cpu().numpy()
    else:
        mask = mask.squeeze()
    
    # Create visualization mask (green channel for lanes)
    mask_viz = np.zeros_like(image)
    mask_viz[:, :, 1] = mask * 255
    
    # Create overlay
    overlay = cv2.addWeighted(image, 0.7, mask_viz, 0.3, 0)
    
    # Stack images horizontally
    result = np.hstack((image, mask_viz, overlay))
    
    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    h, w = image.shape[:2]
    cv2.putText(result, "Original", (10, 30), font, 1, (255, 255, 255), 2)
    cv2.putText(result, "Lane Mask", (w + 10, 30), font, 1, (255, 255, 255), 2)
    cv2.putText(result, "Overlay", (2*w + 10, 30), font, 1, (255, 255, 255), 2)
    
    return result

def main():
    tusimple_config = {
        'json_paths': ["/home/luis_t2/SEAME/TUSimple/train_set/label_data_0313.json",
                      "/home/luis_t2/SEAME/TUSimple/train_set/label_data_0531.json",
                      "/home/luis_t2/SEAME/TUSimple/train_set/label_data_0601.json"],
        'img_dir': '/home/luis_t2/SEAME/TUSimple/train_set/',
        'width': 384,
        'height': 384,
        'is_train': False,
        'thickness': 8
    }

    # Load dataset
    dataset = TuSimpleDataset(**tusimple_config)
    print(f"Loaded Carla dataset with {len(dataset)} samples")
    
    # Visualization loop
    idx = 0
    total_samples = len(dataset)
    
    # Create a fixed window name
    WINDOW_NAME = "TUSimple Dataset Visualization"
    
    print("\nControls:")
    print("  'n' - Next image")
    print("  'p' - Previous image")
    print("  'q' - Quit")
    
    # Create the window once
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    
    while True:
        # Get sample
        image, mask = dataset[idx]
        
        # Create visualization
        vis_image = visualize_sample(image, mask)
        
        # Add sample count to the image instead of the window title
        h, w = vis_image.shape[:2]
        status_text = f"Sample: {idx+1}/{total_samples}"
        cv2.putText(vis_image, status_text, (w//2 - 80, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Show image - using the same window name each time
        cv2.imshow(WINDOW_NAME, cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
        key = cv2.waitKey(0) & 0xFF
        
        if key == ord('q'):  # Quit
            break
        elif key == ord('n'):  # Next
            idx = (idx + 1) % total_samples
        elif key == ord('p'):  # Previous
            idx = (idx - 1) % total_samples
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()