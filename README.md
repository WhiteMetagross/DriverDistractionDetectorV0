# Driver Behavior Distraction Detection:

## Project Overview:

The Driver Behavior Distraction Detection system is a production-grade desktop application designed to analyze and classify driver behavior in real-time. Using advanced machine learning techniques, the system detects driver regions via YOLO, extracts deep features with ResNet50, classifies behavior using XGBoost, and provides visual explanations through Grad-CAM heatmaps overlaid on edge-enhanced frames. The application features a modern PySide6 desktop GUI for seamless user interaction.

## Purpose:

This application aims to enhance road safety by identifying potentially dangerous driver behaviors such as phone usage, passenger interaction, and reaching behind. Real-time monitoring and classification enable timely alerts and intervention, contributing to accident prevention and improved driving safety.

## Key Features:

- **Real-time Processing:** Process video streams from webcams or video files in real-time.
- **YOLO-based Detection:** Accurate detection of driver regions using YOLOv8.
- **Deep Feature Extraction:** ResNet50-based feature extraction for robust behavior representation.
- **XGBoost Classification:** High-performance behavior classification using XGBoost models.
- **Grad-CAM Visualization:** Visual explanations of model predictions with Grad-CAM heatmaps.
- **Edge Enhancement:** Canny edge overlays for improved visual saliency.
- **Modern GUI:** User-friendly PySide6 desktop interface.
- **YAML Configuration:** Flexible configuration management for easy customization.
- **Modular Architecture:** Clean separation of concerns for easy maintenance and extension.

## Project Structure:

```
DriverBehaviorDistractionv1/
+-- driversafety/
|   +-- __init__.py
|   +-- config/
|   |   +-- default.yaml                 # Configuration file
|   |   +-- loader.py                    # Config loader
|   +-- detection/
|   |   +-- face_detector.py             # YOLO-based driver detection
|   +-- classification/
|   |   +-- feature_extractor.py         # ResNet50 feature extraction
|   |   +-- behavior_classifier.py       # XGBoost classifier
|   +-- visualization/
|   |   +-- gradcam.py                   # Grad-CAM utilities
|   |   +-- edges.py                     # Canny edge visualization
|   |   +-- overlays.py                  # Heatmap overlays
|   +-- gui/
|       +-- app.py                       # Application entry point
|       +-- main_window.py               # Main GUI window
|       +-- video_worker.py              # Background processing thread
|       +-- windows/                     # Additional GUI windows
+-- models/
|   +-- yolo/
|   |   +-- yolov8_driver_face_detector.pt
|   +-- xgboost/
|   |   +-- resnet_features_behavior_classifier.json
|   +-- pytorch/
|       +-- (legacy models)
+-- main.py                              # Application launcher
+-- requirements.txt                     # Python dependencies
+-- setup.ps1                            # Setup script (PowerShell)
+-- start.ps1                            # Start script (PowerShell)
+-- InstallationAndSetup.md              # Installation guide
+-- Usage.md                             # Usage documentation
+-- LICENSE.md                           # License information
```

## Behavior Classification Labels:

The system classifies driver behavior into the following categories:

- **0: Safe Drive**: Normal, attentive driving.
- **1: Using Phone**: Driver is holding or using a mobile device.
- **2: Talking on Phone**: Driver is engaged in phone conversation.
- **3: Trying to Reach Behind**: Driver is reaching toward the back seat.
- **4: Talking to Passenger**: Driver is interacting with passengers.

## Start Guide:

### Prerequisites:

- Python 3.11.
- Windows 10 or later (or macOS/Linux).
- At least 8 GB RAM.

### Installation:

1. **Clone or download the repository.**

2. **Run the setup script:**

   ```powershell
   .\setup.ps1
   ```

3. **Ensure model files are present in the `models/` directory.**

### Running the Application:

**Option 1: Using the start script:**

```powershell
.\start.ps1
```

**Option 2: Direct Python execution:**

```bash
python main.py
```

### Using the Application:

1. **Launch the application** using one of the methods above.
2. **Select input source:**
  : Click Start Webcam for real-time webcam processing.
  : Click Open Video File to process a video file.
3. **View results:**
  : Real-time behavior classification.
  : Grad-CAM heatmaps showing model focus areas.
  : Confidence scores for each prediction.
4. **Stop processing** by clicking the Stop button.

## Technical Architecture:

### Detection Pipeline:

The YOLO-based detector identifies the driver region in each frame, producing bounding boxes for subsequent processing.

### Feature Extraction:

ResNet50 (pre-trained on ImageNet) extracts a 2048-dimensional feature vector from the detected driver region.

### Classification:

The XGBoost classifier maps extracted features to behavior classes with high accuracy and interpretability.

### Visualization:

Grad-CAM generates attention maps targeting the final ResNet layer, overlaid on Canny edge-enhanced frames for improved visual clarity.

### GUI:

The PySide6-based GUI runs frame processing in a background thread to maintain UI responsiveness during real-time analysis.

## Configuration:

Edit `driversafety/config/default.yaml` to customize:

- Model file paths.
- Confidence thresholds.
- Input frame dimensions.
- Processing device (CPU/CUDA).

## Performance Benchmarks:

Expected performance on standard hardware:

- **Accuracy:** ~92% on test dataset.
- **Macro-F1 Score:** ~0.90.
- **Latency (CPU):** ~18 ms/frame.
- **Latency (CUDA):** ~9 ms/frame.
- **FPS (CPU):** ~15 FPS @ 960px width.
- **FPS (CUDA):** ~30 FPS @ 960px width.

## Troubleshooting:

For detailed troubleshooting information, see [InstallationAndSetup.md](InstallationAndSetup.md).

## Documentation:

- **[InstallationAndSetup.md](InstallationAndSetup.md):** Comprehensive installation and setup guide.
- **[Usage.md](Usage.md):** Detailed usage instructions and examples.
- **[LICENSE.md](LICENSE.md):** License information.

## License:

This project is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0) License. See [LICENSE.md](LICENSE.md) for details.

## Contributing:

Contributions are welcome. Please ensure code follows the project's modular architecture and includes appropriate documentation.

## Support:

For issues, questions, or suggestions, please refer to the documentation or contact the project maintainers.

## Visual Demonstrations:

### Application Interface:

![Driver Safety Window with Performance Metrics and Visualization](visuals/DriverDafetyWindowWithPerformanceMetricsAndVisualization.png)

The main application window displays real-time video processing with overlaid behavior classification results, Grad-CAM heatmaps, and performance metrics including FPS and confidence scores.

### Sample Video: Asian Driver Talking on Phone

[View video: AsianDriverTalkingOnPhone.mp4](visuals/AsianDriverTalkingOnPhone.mp4)

Demonstrates real-time detection and classification of a driver engaged in phone conversation, showcasing the model's ability to identify the distracting behavior with appropriate confidence scoring.

### Sample Video: Female Driver with Normal Driving

[View video: FemaleDriverWithNormalDriving.mp4](visuals/FemaleDriverWithNormalDriving.mp4)

Demonstrates normal, attentive driving behavior classification, showing how the system correctly identifies safe driving patterns and maintains low confidence scores for non-distracting behaviors.

## Acknowledgments:

This project leverages state-of-the-art machine learning frameworks and models:

- **YOLO:** Ultralytics YOLOv8 for object detection.
- **ResNet:** PyTorch ResNet50 for feature extraction.
- **XGBoost:** Gradient boosting for classification.
- **Grad-CAM:** Visual explanations for model predictions.
- **PySide6:** Modern desktop GUI framework.
