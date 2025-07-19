from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QTextBrowser


class LogBrowser(QTextBrowser):
    def __init__(self, parent=None):
        super().__init__(parent)

    def keyPressEvent(self, event):
        # The original function of arrow keys is disabled
        if event.key() in [Qt.Key_Up, Qt.Key_Down, Qt.Key_Left, Qt.Key_Right]:
            event.ignore()
        else:
            super().keyPressEvent(event)