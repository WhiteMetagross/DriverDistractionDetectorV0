# Models Directory

This folder organizes all model artifacts used by the Driver Safety Detection project.

## Layout
- yolo/
  - yolov8_driver_face_detector.pt — YOLOv8 detector weights used to locate the driver/face region in frames. Provided by you.
- xgboost/
  - resnet_features_behavior_classifier.json — Primary XGBoost classifier trained on ResNet50 features to classify driver behavior.
  - xg_driver.json — Optional alternate XGBoost classifier (older/alternate variant; purpose relative to the primary is unclear; kept as-is).
- pytorch/
  - legacy_pytorch_driver_activity.pth — Legacy PyTorch checkpoint, likely a driver activity classifier from an earlier prototype; not used by the current pipeline.
  - legacy_pytorch_checkpoint_4.pth — Legacy PyTorch checkpoint (unknown purpose); not used by the current pipeline.

## Notes
- The current pipeline uses ImageNet-pretrained ResNet50 for feature extraction (torchvision weights), not a custom PyTorch .pth file.
- If you'd like to switch to a finetuned PyTorch backbone, place the .pth file here and add a config entry + loading logic.
- Training dates/metrics were not provided for the legacy .pth files; if you know them, please update this README with details.

