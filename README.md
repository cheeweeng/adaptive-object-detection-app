# Adaptive Object Detection App

## Overview
The Adaptive Object Detection App is a zero-shot object detection system built for surveillance applications. It leverages Google's OWL-ViT (Zero-Shot Object Detection Transformer) to count and identify objects in surveillance footage using natural language prompts, without requiring model retraining.

## Key Features
- **Zero-shot detection**: Detect any object by simply describing it in natural language
- **Real-time annotation**: Visualize detected objects with bounding boxes and confidence scores
- **Object counting**: Automatically count detected objects and generate summary statistics
- **Adjustable confidence threshold**: Fine-tune detection sensitivity
- **Multiple object support**: Detect and count multiple object categories simultaneously
- **GPU acceleration**: Optimized for NVIDIA GPUs with CUDA support

## Technical Stack
- **Framework**: PyTorch with CUDA support
- **Model**: Google OWL-ViT Base (`google/owlvit-base-patch32`)
- **UI**: Gradio for interactive web interface
- **Libraries**: Transformers, Pillow, OpenCV, NumPy
- **Language**: Python 3.10+

## Project Structure
```
.
├── CLAUDE.md                # Development guidance for Claude Code
├── README.md                # Project documentation
├── Prompt used in Claude.txt # Project description
├── owlvit_surveillance.ipynb # Main Jupyter notebook with implementation
└── VGAP_OWLViT_Report.docx  # Project report
```

## Getting Started

### Prerequisites
- Python 3.10+
- NVIDIA GPU with CUDA support (recommended for performance)
- Basic knowledge of Jupyter notebooks and Gradio

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/<your-username>/adaptive-object-detection-app.git
   cd adaptive-object-detection-app
   ```

2. **Install dependencies**
   ```bash
   # Install PyTorch with CUDA support
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

   # Install other required packages
   pip install transformers gradio Pillow opencv-python
   ```

### Running the Application

1. **Start Jupyter Notebook**
   ```bash
   jupyter notebook
   ```

2. **Open `owlvit_surveillance.ipynb`**

3. **Run all cells** (Cell → Run All) to launch the Gradio interface.

4. **Access the app** at `http://127.0.0.1:7860`

### Usage

- Upload a surveillance image
- Enter comma-separated object labels to detect (e.g., `car, truck, bus`)
- Adjust the confidence threshold as needed
- Click "Detect & Count" to see annotated results and object counts

### Demo  
<img width="812" height="352" alt="image" src="https://github.com/user-attachments/assets/eca1d11e-2b58-4d58-9430-2d813359df82" />  

The application was tested using a parking lot image containing multiple vehicles. The key advantage of zero-shot detection is that the model can identify objects based on text prompts provided by the user, rather than relying on a fixed list of predefined categories. For instance, I have entered “car” as the prompt and the output  showed that the zero-shot object detector successfully identified and generated accurate bounding boxes of the cars. The successful detection result demonstrated that the application could effectively analyze uploaded images and provide reliable object identification and counting functionality by using pre-trained generative AI models to build intelligent computer vision applications without training a model from scratch.  

## Future Improvements

- **Enhance Detection Speed**: Optimize the OWL-ViT model for real-time processing on lower-end hardware.
- **Expand Detection Categories**: Add support for more specific or niche object categories (e.g., construction equipment, wildlife).
- **Real-Time Video Processing**: Extend the app to process video streams from cameras or predefined folders.
- **Advanced UI Enhancements**: Improve the Gradio interface with filters, animations, or export options for annotated images.
- **Custom Model Training**: Provide an option to fine-tune the model on domain-specific data for improved accuracy.
- **Integration with Different Sources**: Allow input from various surveillance sources (e.g., IP cameras, local files).
- **Cross-Platform Support**: Ensure compatibility with Windows, macOS, and Linux without CUDA dependencies.

## Contributing
We welcome contributions! Please fork the repository and submit a pull request with your changes.

## License
This project is licensed under the MIT License - see the `LICENSE` file for details.

## Acknowledgments
- Google Research for OWL-ViT model
- Hugging Face for Transformers library
- Gradio for the intuitive UI framework
