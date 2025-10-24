# Installation and Setup:

## System Requirements:

- **Operating System:** Windows 10 or later, macOS, or Linux.
- **Python Version:** Python 3.11.
- **RAM:** Minimum 8 GB (16 GB recommended for optimal performance).
- **GPU (Optional):** NVIDIA GPU with CUDA support for accelerated processing.
- **Disk Space:** At least 5 GB for dependencies and model files.

## Prerequisites:

Before starting the installation, ensure you have the following installed on your system:

1. **Python 3.11:** Download from [https://www.python.org/](https://www.python.org/).
2. **Git (Optional):** For cloning the repository from [https://git-scm.com/](https://git-scm.com/).
3. **pip:** Usually comes with Python; verify by running `pip --version`.

## Installation Steps:

### Step 1: Clone or Download the Repository:

If using Git:

```bash
git clone <repository-url>
cd DriverBehaviorDistractionv1
```

Or download the repository as a ZIP file and extract it to your desired location.

### Step 2: Run the Setup Script (Recommended):

On Windows (PowerShell):

```powershell
.\setup.ps1
```

This script will:
- Check Python installation and verify Python 3.11.
- Create a virtual environment named `driver_safety` (if not already present).
- Activate the virtual environment.
- Upgrade pip.
- Install all dependencies from `requirements.txt`.
- Install PyTorch and torchvision with CUDA 12.1 support.
- Verify model files.

**Optional flags:**
- `-SkipVenv`: Skip virtual environment creation and activation.

### Step 3: Manual Setup (Alternative):

If you prefer to set up manually or the script fails:

1. **Create a virtual environment:**

   ```bash
   python -m venv driver_safety
   ```

2. **Activate the virtual environment:**

   - Windows (PowerShell):
     ```powershell
     .\driver_safety\Scripts\Activate.ps1
     ```

   - Windows (Command Prompt):
     ```cmd
     driver_safety\Scripts\activate.bat
     ```

   - macOS/Linux:
     ```bash
     source driver_safety/bin/activate
     ```

3. **Upgrade pip:**

   ```bash
   python -m pip install --upgrade pip
   ```

4. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

### Step 4: Verify Model Files:

Ensure the following model files are present in the `models/` directory:

- `models/yolo/yolov8_driver_face_detector.pt` - YOLOv8 detector for driver/face region.
- `models/xgboost/resnet_features_behavior_classifier.json` - XGBoost behavior classifier.

If these files are missing, the application will not function correctly.

### Step 5: Install CUDA 12.1 Supported PyTorch (If Needed):

For GPU acceleration and CUDA 12.1 support, install or repair PyTorch and torchvision using the official CUDA 12.1 index URL.

```bash
pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cu121 --upgrade --force-reinstall
```

Visit [https://pytorch.org/](https://pytorch.org/) for the latest installation instructions and to confirm commands for your environment.

## Configuration:

The application uses a YAML configuration file located at `driversafety/config/default.yaml`.

Common configuration options:

- **Model paths:** Paths to YOLO and XGBoost model files.
- **Confidence threshold:** Minimum confidence for detections.
- **Resize width:** Input frame width for processing.
- **Device:** CPU or CUDA device selection.

Edit this file to customize the application behavior.

## Troubleshooting:

### Issue: Python not found in PATH.

**Solution:** Ensure Python is installed and added to your system PATH. Reinstall Python and check "Add Python to PATH" during installation.

### Issue: Virtual environment activation fails.

**Solution:** On Windows, you may need to change the execution policy:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Issue: Dependencies fail to install.

**Solution:** Ensure you have an active internet connection and sufficient disk space. Try upgrading pip first:

```bash
python -m pip install --upgrade pip
```

### Issue: Model files not found.

**Solution:** Verify that model files are in the correct directories. Check the paths in `driversafety/config/default.yaml`.

### Issue: Qt plugin errors on startup.

**Solution:** The setup script automatically configures Qt paths. If issues persist, manually set the Qt plugin path:

```powershell
$env:QT_PLUGIN_PATH = "C:\path\to\python\Lib\site-packages\PySide6\plugins"
```

## Next Steps:

After successful installation, proceed to the [Usage.md](Usage.md) file to learn how to use the application.

For more information about the project structure and technical details, see [README.md](README.md).

