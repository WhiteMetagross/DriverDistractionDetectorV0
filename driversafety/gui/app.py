import sys
import os
from PySide6.QtWidgets import QApplication
from driversafety.gui.main_window import MainWindow


def setup_qt_plugins():
    """
    Set up Qt plugin path to fix platform plugin issues.
    This ensures Qt can find the necessary platform plugins on Windows.
    """
    try:
        import PySide6
        pyside_dir = os.path.dirname(PySide6.__file__)
        plugins_dir = os.path.join(pyside_dir, 'plugins')

        if os.path.exists(plugins_dir):
            # Set the Qt plugin path environment variable
            os.environ['QT_PLUGIN_PATH'] = plugins_dir
            print(f"Qt plugin path set to: {plugins_dir}")
        else:
            print(f"Warning: Qt plugins directory not found at {plugins_dir}")
    except ImportError:
        print("Warning: PySide6 not found, cannot set Qt plugin path")


def main():
    # Set up Qt plugins before creating QApplication
    setup_qt_plugins()

    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

