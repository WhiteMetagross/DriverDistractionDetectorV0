from PySide6.QtGui import QColor

# Simple design system
PRIMARY = QColor(20, 108, 148)  # teal-ish
ACCENT = QColor(232, 93, 117)   # coral
OK = QColor(46, 204, 113)       # green
WARN = QColor(241, 196, 15)     # yellow
ERROR = QColor(231, 76, 60)     # red
TEXT = QColor(33, 33, 33)
BG = QColor(248, 249, 250)

CLASS_COLORS = {
    0: OK,    # safe drive
    1: ERROR, # Using phone
    2: ERROR, # Talking on phone
    3: WARN,  # Trying to reach behind
    4: WARN,  # Talking to a passenger
}

APP_STYLESHEET = """
QMainWindow, QWidget {
    background-color: #f8f9fa;
    color: #212121;
    font-family: Segoe UI, Roboto, Arial;
    font-size: 12pt;
}
QPushButton {
    padding: 8px 14px;
    border-radius: 6px;
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                stop:0 #2e86c1, stop:1 #1f618d);
    color: white;
}
/* Qt-compatible hover without CSS filter */
QPushButton:hover {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                stop:0 #3690cf, stop:1 #2a6ea6);
}
QPushButton:pressed {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                stop:0 #1f618d, stop:1 #154360);
}
QPushButton:disabled { background: #bdc3c7; }
QLabel#Title { font-size: 16pt; font-weight: 600; }
QStatusBar { background: #ffffff; border-top: 1px solid #e0e0e0; }
"""

# Optional dark theme
DARK_STYLESHEET = """
QMainWindow, QWidget {
    background-color: #1e1e1e;
    color: #e0e0e0;
    font-family: Segoe UI, Roboto, Arial;
    font-size: 12pt;
}
QPushButton {
    padding: 8px 14px;
    border-radius: 6px;
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                stop:0 #34495e, stop:1 #2c3e50);
    color: #f5f5f5;
}
QPushButton:hover {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                stop:0 #3e5871, stop:1 #34495e);
}
QPushButton:pressed {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                stop:0 #2c3e50, stop:1 #22303d);
}
QPushButton:disabled { background: #5c6b73; color: #9aa7ad; }
QLabel#Title { font-size: 16pt; font-weight: 600; }
QStatusBar { background: #2b2b2b; border-top: 1px solid #3a3a3a; }
"""

def get_stylesheet(mode: str = "light") -> str:
    return DARK_STYLESHEET if mode.lower() == "dark" else APP_STYLESHEET

