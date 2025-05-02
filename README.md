# Lane Detection with Deep Learning

## Project Overview
This repository contains a deep learning-based lane detection system designed to identify lane markings on roads from video input. The system uses semantic segmentation techniques with various U-Net architectures to detect and highlight lane boundaries.

## Features
- Multiple model architectures (UNet, MobileNetV2UNet, LightUNet)
- Support for multiple datasets (TuSimple, SEAME, Carla)
- Weighted dataset sampling for balanced training
- Advanced image augmentation pipeline
- Real-time inference on video streams
- ONNX model export for deployment

## Model Architectures
The project implements three different U-Net variants:
- **UNet**: Standard U-Net architecture with 64 base filters  
- **MobileNetV2UNet**: U-Net with MobileNetV2 backbone for more efficient processing  
- **LightUNet**: Lightweight U-Net with 32 base filters for resource-constrained environments

## Datasets
The model supports training on multiple datasets:
- **TuSimple**: Large-scale lane detection dataset with annotated lane markings
- **SEAME**: Custom dataset for specific road conditions
- **Carla**: Synthetic data generated from the CARLA simulator

A `CombinedDataset` class provides functionality to merge multiple datasets with configurable weights.

## Training
Training is performed using:
- Binary Cross-Entropy With Logits Loss for lane segmentation
- Adam optimizer
- Extensive data augmentation using `albumentations`
- Weighted sampling to balance dataset contributions

To train the model:
```bash
python main.py
````

The training script saves model checkpoints after each epoch.

## Inference

The repository includes code for running inference on videos:

```bash
python inference.py
```

This processes video input and displays the detected lanes overlaid on the original frames.

## Model Conversion

For deployment on other platforms, models can be exported to ONNX format:

```bash
python convert.py
```

## Project Structure

```
.
├── assets/                  # Video files for testing
├── Models/                  # Saved model weights
│   └── lane/                # Lane detection models
├── src/                     # Source code
│   ├── augmentation.py      # Data augmentation pipeline
│   ├── CarlaDataset.py      # Carla dataset handler
│   ├── CombinedDataset.py   # Combined dataset handler
│   ├── SEAMEDataset.py      # SEAME dataset handler
│   ├── train.py             # Training functions
│   ├── TUSimpleDataset.py   # TuSimple dataset handler
│   └── unet.py              # Model architectures
├── convert.py               # ONNX conversion script
├── inference.py             # Run inference on videos
├── main.py                  # Main training script
└── requirements.txt         # Project dependencies
```

## Requirements

This project requires the following dependencies:

```
torch
torchvision
timm
matplotlib
pandas
numpy
opencv-python
albumentations
onnx
```

Install all requirements with:

```bash
pip install -r requirements.txt
```

## Usage

### Training

1. Configure dataset paths in `main.py`
2. Select the desired model architecture
3. Run:

```bash
python main.py
```

### Inference

1. Make sure you have a trained model in the `Models/lane/` directory
2. Specify the video file in `inference.py`
3. Run:

```bash
python inference.py
```

## Acknowledgements

* TuSimple for providing the lane detection dataset
* CARLA Simulator for synthetic data generation

