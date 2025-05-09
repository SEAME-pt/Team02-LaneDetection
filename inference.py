import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import numpy as np
import cv2
from torchvision import transforms
from src.unet import UNet
import time

# Set up device
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Using CUDA device: {torch.cuda.get_device_name()}")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS (Metal Performance Shaders)")
else:
    device = torch.device("cpu")
    print("Using CPU")

# Load the trained model
model = UNet().to(device)
model.load_state_dict(torch.load('Models/lane/lane_UNet1_epoch_25.pth', map_location=device))
model.eval()


src_pts = np.float32([
    [248.0, 81.0],
    [394.0, 81.0],
    [32.0, 456.0],
    [608.0, 456.0],
])

# Image preprocessing function
def preprocess_image(image, target_size=(256, 128)):
    # Resize image
    img = cv2.resize(image, target_size)
    
    # 2. Enhance contrast within the ROI
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Apply same transforms as during training
    normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([
        transforms.ToTensor(),
        normalizer
    ])
    
    # Apply transforms
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    return img_tensor, img
# Function to overlay lane predictions on image
def overlay_predictions(image, prediction, threshold=0.6):
    # Convert prediction to binary mask
    prediction = prediction.squeeze().cpu().detach().numpy()
    lane_mask = (prediction > threshold).astype(np.uint8) * 255
    
    # Resize mask to match the original image size
    lane_mask = cv2.resize(lane_mask, (image.shape[1], image.shape[0]))
    
    # Create a colored overlay
    colored_mask = np.zeros_like(image)
    colored_mask[lane_mask > 0] = [0, 255, 0]  # Green for lane markings

    image, _ = get_bird_eye_view(image, None, src_pts=src_pts, extended_view=True)
    _, colored_mask = get_bird_eye_view(None, colored_mask, src_pts=src_pts, extended_view=True)
    
    # Apply the overlay with transparency
    overlay = cv2.addWeighted(image, 0.7, colored_mask, 0.3, 0)
    return overlay
    

def get_bird_eye_view(image, mask=None, src_pts=None, dst_pts=None, extended_view=True):
    """
    Transform image and/or mask to bird's eye view with extended road visibility
    """
    h, w = image.shape[:2] if image is not None else mask.shape[:2]
    
    # Default source points if not provided
    if src_pts is None:
        src_pts = np.float32([
            [w * 0.40, h * 0.45],  # Top left (higher up)
            [w * 0.60, h * 0.45],  # Top right (higher up)
            [w * 0.05, h * 0.95],  # Bottom left
            [w * 0.95, h * 0.95]   # Bottom right
        ])
    
    # Create non-linear transformation for better distance perception
    if dst_pts is None:
        bev_width, bev_height = w, h
        margin = int(bev_width * 0.1)
        
        dst_pts = np.float32([
            [bev_width * 0.25, 0],                # Top left (closer to center)
            [bev_width * 0.75, 0],                # Top right (closer to center) 
            [margin, bev_height],                 # Bottom left
            [bev_width - margin, bev_height]      # Bottom right
        ])

    # Compute the perspective transform matrix
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    
    # Apply the transform
    bev_image = None
    if image is not None:
        bev_image = cv2.warpPerspective(image, M, (w, h), flags=cv2.INTER_LINEAR)
    
    bev_mask = None
    if mask is not None:
        bev_mask = cv2.warpPerspective(mask, M, (w, h), flags=cv2.INTER_NEAREST)
    
    return bev_image, bev_mask


# Open video
cap = cv2.VideoCapture("assets/seame_data.mp4")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    time.sleep(0.05)  # Optional: Add a small delay to control frame rate
    
    # Preprocess the image
    img_tensor, original_frame = preprocess_image(frame)
    
    # Run inference
    with torch.no_grad():
        predictions = model(img_tensor)
        predictions = torch.sigmoid(predictions)
    
    # Overlay predictions on the original frame
    result_frame = overlay_predictions(frame, predictions)
    
    # Display the result
    cv2.imshow("Lane Detection", result_frame)
    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
