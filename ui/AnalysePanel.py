import os
import platform
import time
from datetime import datetime
from threading import Thread

import numpy as np
import onnxruntime
import scipy
from PyQt5 import QtGui
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import QFrame, QPushButton, QVBoxLayout, QLabel, QTextEdit, QTextBrowser, QHBoxLayout, QFileDialog

from ui.LogBrowser import LogBrowser


class AnalysePanel(QFrame):
    signal = pyqtSignal(str)

    def __init__(self, csi, config, parent=None):
        super().__init__(parent)
        self.csi = csi
        self.config = config
        self.parentWindow = parent
        self.analysing = False
        self.loaded = False

        self.initUI()
        if hasattr(self.config, "ort_session"):
            self.loaded = True
            self.analyseBtn.setDisabled(False)
            self.addLog("Model has been loaded automatically!", level="success")



    def initUI(self):
        self.loadBtn = QPushButton('load model')
        self.loadBtn.clicked.connect(self.loadBtnClick)

        self.analyseBtn = QPushButton('start analysis')
        self.analyseBtn.clicked.connect(self.analyseBtnClick)
        self.analyseBtn.setDisabled(True)

        hBox = QHBoxLayout()
        hBox.addStretch(1)
        hBox.addWidget(self.loadBtn)
        hBox.addStretch(1)
        hBox.addWidget(self.analyseBtn)
        hBox.addStretch(1)

        self.logBox = LogBrowser()

        vBox = QVBoxLayout()
        # vBox.setContentsMargins(0, 10, 0, 0)
        vBox.addLayout(hBox)
        vBox.addWidget(self.logBox)
        vBox.setContentsMargins(0, 10, 0, 0)  # Set the inside and outside margins to 0

        self.setLayout(vBox)
        self.setFrameShape(QFrame.Panel)  # Set panel Shape
        self.setLineWidth(1)  # Set the border line width

    def loadBtnClick(self):
        self.addLog("Please select the file of ONNX model ...")
        modelPath = QFileDialog.getOpenFileName(self, "choose file", "./", "ONNX Files (*.onnx)")[0]
        if not modelPath:
            self.addLog("No file was chosen!", level="error")
            return
        try:
            self.config.ort_session = onnxruntime.InferenceSession(modelPath, providers=['CPUExecutionProvider'])
        except Exception as e:
            print(e)
            self.addLog("error:"+str(e), level="error")
            return
        self.loaded = True
        self.analyseBtn.setDisabled(False)
        self.addLog("Model loading success!", level="success")

    def analyseBtnClick(self):
        if not self.analysing:
            self.detection_start()
        else:
            self.detection_stop()

    def detection_start(self):
        self.analysing = True
        self.thread = Thread(target=self.process, daemon=True)
        self.thread.start()

        self.analyseBtn.setText("stop analysis")
        self.signal.emit("analyse start")

    def detection_stop(self):
        self.analysing = False
        self.thread.join()

        self.analyseBtn.setText("start analysis")
        self.signal.emit("analyse stop")

    def process(self):
        while self.analysing:
            csi = self.csi.copy()
            addLog = self.addLog
            self.config.detection(csi, addLog)
            time.sleep(0.2)

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
        self.logBox.moveCursor(QtGui.QTextCursor.End)
