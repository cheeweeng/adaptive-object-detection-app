# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a surveillance object detection system built using OWL-ViT (Zero-Shot Object Detection) and Gradio. The system allows users to count different objects (vehicles, pedestrians, animals) in surveillance camera footage using natural language prompts without retraining the model.

## Architecture

### Main Components
- **Jupyter Notebook (`owlvit_surveillance.ipynb`)**: Contains the complete implementation
- **Model**: Google's OWL-ViT Base model (`google/owlvit-base-patch32`)
- **Framework**: PyTorch with CUDA support
- **UI**: Gradio web interface

### Key Features
- Zero-shot object detection using text prompts
- Real-time annotation of detected objects
- Object counting with summary statistics
- Adjustable confidence threshold
- Support for multiple object categories in a single prompt

## Development Commands

### Running the Application
```bash
# Start Jupyter notebook
jupyter notebook

# The app launches on http://127.0.0.1:7860
# To create a public link, modify demo.launch(share=True)
```

### Dependencies Installation
```bash
# Install PyTorch with CUDA support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install required packages
pip install transformers gradio Pillow opencv-python
```

### GPU Check
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
```

## Code Structure

### Main Functions
1. `detect_objects()`: Core detection function using OWL-ViT
2. `annotate_image()`: Draws bounding boxes and creates count summary
3. `run_pipeline()`: Gradio pipeline connecting inputs to outputs
4. `get_color()`: Color palette for different object types

### Input Format
- Image upload via Gradio interface
- Text prompt with comma-separated object labels (e.g., "car, truck, bus")
- Confidence threshold slider (0.05-0.95)

### Output
- Annotated image with colored bounding boxes
- Text summary showing object counts

## Important Notes

- The model uses unauthenticated Hugging Face Hub access
- First run may show warning about HF_TOKEN (can be ignored for development)
- Objects are detected using zero-shot classification with natural language descriptions
- The system applies NMS (Non-Maximum Suppression) to remove overlapping detections