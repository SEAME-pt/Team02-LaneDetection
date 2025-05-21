import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import numpy as np
import cv2
from torchvision import transforms
from src.unet import UNet, MobileNetV2UNet
from src.YoloSeg import YOLOPSeg
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
model = YOLOPSeg().to(device)
model.load_state_dict(torch.load('Models/lane/lane_Mob2_epoch_12.pth', map_location=device))
model.eval()

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
    
def post_process(lane_mask, kernel_size=10, min_area=100, max_lanes=6):
    """
    Color lanes based on their horizontal position: left=red, center=yellow, right=green
    """
    # Ensure input is binary uint8 image
    if lane_mask.dtype is not np.uint8:
        lane_mask = np.array(lane_mask, np.uint8)
    if len(lane_mask.shape) == 3:
        lane_mask = cv2.cvtColor(lane_mask, cv2.COLOR_BGR2GRAY)

    # Create a colored mask (3-channel)
    colored_lanes = np.zeros((lane_mask.shape[0], lane_mask.shape[1], 3), dtype=np.uint8)

    # Fill the pixel gap using Closing operator (dilation followed by erosion)
    kernel = cv2.getStructuringElement(shape=cv2.MORPH_RECT, ksize=(kernel_size, kernel_size))

    lane_mask = cv2.morphologyEx(lane_mask, cv2.MORPH_CLOSE, kernel)
    
    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        lane_mask, connectivity=8, ltype=cv2.CV_32S)
    
    # Define standard colors for lane positions
    lane_position_colors = [
        [0, 0, 255],      # Far left: Red
        [0, 128, 255],    # Left: Orange
        [0, 255, 255],    # Center left: Yellow
        [0, 255, 0],      # Center right: Green
        [255, 0, 0],      # Right: Blue
        [255, 0, 255],    # Far right: Purple
    ]
    
    # Create a list of valid components with their centroids
    valid_components = []
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area > min_area:
            # Get centroid x-position
            center_x = centroids[i][0]
            valid_components.append((i, center_x))

    # Sort components by area (largest first)
    area_sorted = sorted(valid_components, key=lambda x: x[1])
    # Keep only the largest max_lanes components
    keep_components = area_sorted[:max_lanes]
    
    # For storing lane polylines
    lane_polylines = []
    
    # Process each lane
    for idx, (comp_idx, _) in enumerate(keep_components):
        # Get this lane's mask
        lane_mask = (labels == comp_idx).astype(np.uint8) * 255
        
        # Color fill the lane in the colored mask
        lane = (labels == comp_idx)
        color = lane_position_colors[min(idx, len(lane_position_colors)-1)]
        colored_lanes[lane] = [0, 255, 0]
        
        # Extract lane coordinates for polyline
        # First, find contours to get a rough outline
        # contours, _ = cv2.findContours(lane_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
        # if contours:
        #     largest_contour = max(contours, key=cv2.contourArea)
            
        #     # Extract main points to represent the lane
        #     lane_points = []
        #     h, w = lane_mask.shape
            
        #     # Sample points from top to bottom (every 5 pixels)
        #     for y in range(0, h, 5):
        #         # Find all points at this y-level
        #         x_points = np.where(lane_mask[y, :] > 0)[0]
        #         if len(x_points) > 0:
        #             # Use the middle point at this y-level
        #             mid_x = (np.min(x_points) + np.max(x_points)) // 2
        #             lane_points.append([mid_x, y])
            
        #     if lane_points:
        #         lane_polyline = np.array(lane_points)
        #         lane_polylines.append((lane_polyline, color))
                
        #         # Draw the polyline on the colored mask
        #         cv2.polylines(
        #             img=colored_lanes,
        #             pts=[lane_polyline],
        #             isClosed=False,
        #             color=color,
        #             thickness=5)
    
    return colored_lanes
    
# Update your overlay_predictions function
def overlay_predictions(image, prediction, threshold=0.5):
    # Convert prediction to binary mask
    prediction = prediction.squeeze().cpu().detach().numpy()
    lane_mask = (prediction > threshold).astype(np.uint8) * 255
    
    # Resize mask to match the original image size
    lane_mask_resized = cv2.resize(lane_mask, (image.shape[1], image.shape[0]))
    
    # Apply lane connection post-processing
    colored_mask = post_process(lane_mask_resized)

    # Apply the overlay with transparency
    overlay = cv2.addWeighted(image, 0.7, colored_mask, 0.3, 0)
    return overlay

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
