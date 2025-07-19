import platform
import time
from datetime import datetime

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QTextCharFormat, QColor, QTextCursor
from PyQt5.QtWidgets import QFrame, QPushButton, QVBoxLayout, QLabel, QSizePolicy, QMessageBox, QFileDialog, QTextEdit, \
    QHBoxLayout, QTextBrowser

from Playback import Playback
from SocketClient import SocketClient
from ui.LogBrowser import LogBrowser


class SignalPanel(QFrame):
    signal = pyqtSignal(str)

    def __init__(self, bfee_list, config, parent=None):
        super().__init__(parent)
        self.bfee_list = bfee_list
        self.config = config
        self.parentWindow = parent
        self.playFile = None
        self.monitoring = False
        self.playing = False
        self.initUI()

    def initUI(self):
        self.monitorBtn = QPushButton('start monitor')
        self.monitorBtn.clicked.connect(self.monitorBtnClick)

        self.playBtn = QPushButton('start playback')
        self.playBtn.clicked.connect(self.playBtnClick)

        hBox = QHBoxLayout()
        hBox.addWidget(self.monitorBtn, alignment=Qt.AlignHCenter)
        hBox.addWidget(self.playBtn, alignment=Qt.AlignHCenter)

        self.logBox = LogBrowser()

        vBox = QVBoxLayout()
        vBox.addLayout(hBox)
        vBox.addWidget(self.logBox)
        vBox.setContentsMargins(0, 10, 0, 0)

        self.setLayout(vBox)
        self.setFrameShape(QFrame.Panel)
        self.setLineWidth(1)

    def monitorBtnClick(self):
        plat = platform.system().lower()
        if plat == 'windows':
            # QMessageBox.critical(self, "Error", "Listen for real-time data on Linux systems only！")
            self.addLog("Listen for real-time data on Linux systems only", level="error")
            return
        if not self.monitoring:
            self.addLog("monitor start...")
            self.monitoring = True
            self.monitorBtn.setText("stop monitor")
            self.signal.emit("monitor started")

            self.socket_client = SocketClient(self.bfee_list, parent=self)
            self.socket_client.finished.connect(self.monitorEnd)
            self.socket_client.start()
            self.updateStatus()
        else:
            self.socket_client.stop()

    def monitorEnd(self):
        self.addLog("monitor closed!")
        self.monitoring = False
        self.monitorBtn.setText("start monitor")
        self.signal.emit("monitor stopped")
        self.updateStatus()

    def playBtnClick(self):
        if not self.playing:
            self.playFile = QFileDialog.getOpenFileName(self, "choose file", "./", "Dat Files (*.dat);;All Files (*)")[0]
            if not self.playFile:
                # QMessageBox.critical(self, "error", "No file was chosen！")
                self.addLog("No file was chosen!", level="error")
                return
            self.addLog("The file was start to playback：" + self.playFile)
            self.playing = True
            self.playBtn.setText("stop playback")
            self.signal.emit("play start")

            self.playback = Playback(self.playFile, self.bfee_list, self.config.fs)
            self.playback.finished.connect(self.playEnd)
            self.playback.start()
            self.updateStatus()
        else:
            # Just send the stop signal
            self.playback.stop()

    def playEnd(self):
        self.signal.emit("stop playback")
        self.addLog("playback complete!")
        self.playing = False
        self.playBtn.setText("start playback")
        self.updateStatus()

    def updateStatus(self):
        if self.monitoring:
            self.playBtn.setDisabled(True)
        else:
            self.playBtn.setDisabled(False)
        if self.playing:
            self.monitorBtn.setDisabled(True)
        else:
            self.monitorBtn.setDisabled(False)

    def addLog(self, msg, level="info"):
        t = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        msg = msg.replace("\n", "<br>")
        self.logBox.append('<font color="#999">' + t + '<font color="red">')
        if level == "info":
            self.logBox.append('<font color="#1E90FF">' + msg + '</font><br>')
        elif level == "warning":
            self.logBox.append('<font color="#FF8C00">' + msg + '</font><br>')
        elif level == "success":
            self.logBox.append('<font color="#87D068">' + msg + '</font><br>')
        elif level == "error":
            self.logBox.append('<font color="#DC143C">' + msg + '</font><br>')
