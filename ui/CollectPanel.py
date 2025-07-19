import glob
import os
import struct
import time
from datetime import datetime

import numpy as np
from PyQt5.QtCore import Qt, pyqtSignal, QObject, QEvent
from PyQt5.QtWidgets import QFrame, QPushButton, QVBoxLayout, QLabel, QHBoxLayout, QLineEdit, QSpinBox, QTextEdit, \
    QTextBrowser, QCheckBox
from numba.typed import List
from scipy.io import savemat

from ui.LogBrowser import LogBrowser
from wifilib_numba_v3 import analyse_bf


class CollectPanel(QFrame):
    signal = pyqtSignal(str, str)

    def __init__(self, config, parent=None):
        super().__init__(parent)
        self.config = config
        self.csi_timestamp = 0
        self.parentWindow = parent

        self.fileName = ""
        self.preName = "./dat/a_0_0_"
        self.fileNum = 0
        self.collecting = False
        self.f_csv = None
        self.f_byte = None
        self.f_event = None
        self.amplitudes = []
        self.pos = 0

        self.save_type = {
            "dat": True,
            "csv": True,
            "npy": True,
            "mat": True
        }

        self.initUI()
        self.refreshFileName()

    def initUI(self):
        # File name
        label = QLabel("file name：")
        min_width = label.fontMetrics().boundingRect(label.text()).width()
        label.setMinimumWidth(min_width + 20)
        self.edit = QLineEdit(self.preName)
        self.edit.setPlaceholderText("prefix")
        self.edit.setToolTip("The prefix of the file")
        self.edit.textChanged.connect(self.onPreChanged)
        self.edit.setMinimumWidth(100)
        self.spinbox = QSpinBox()
        self.spinbox.setMaximum(999)
        self.spinbox.setToolTip("The serial number of the file")
        self.spinbox.valueChanged.connect(self.onNumChanged)
        self.spinbox.setMinimumWidth(50)

        self.collectBtn = QPushButton('start collection')
        self.collectBtn.clicked.connect(self.collectBtnClick)

        hBox = QHBoxLayout()
        hBox.addWidget(label)
        hBox.addStretch(3)
        hBox.addWidget(self.edit)
        hBox.addWidget(self.spinbox)
        hBox.addWidget(self.collectBtn, alignment=Qt.AlignHCenter)
        hBox.setSpacing(0)

        self.checkboxList = QHBoxLayout()
        label2 = QLabel("file type:")
        min_width = label.fontMetrics().boundingRect(label.text()).width()
        label2.setMinimumWidth(min_width + 20)
        self.checkboxList.addWidget(label2)
        self.checkboxList.addStretch(1)
        for item in ["dat", "csv", "npy", "mat"]:
            checkbox = QCheckBox(item)
            if item in self.config.collect_types:
                checkbox.setChecked(True)
                self.save_type[item] = True
            else:
                checkbox.setChecked(False)
                self.save_type[item] = False
            checkbox.stateChanged.connect(self.onCheckBoxChanged)
            self.checkboxList.addWidget(checkbox)

        self.logBox = LogBrowser()

        vBox = QVBoxLayout()
        vBox.addLayout(hBox)
        vBox.addLayout(self.checkboxList)
        # vBox.addWidget(self.collectBtn, alignment=Qt.AlignHCenter)
        vBox.addWidget(self.logBox)
        vBox.setContentsMargins(0, 10, 0, 0)

        self.setLayout(vBox)
        self.setFrameShape(QFrame.Panel)
        self.setLineWidth(1)

    def onPreChanged(self, s):
        if self.preName != s:
            self.preName = s
            self.refreshFileName()

    def onNumChanged(self, s):
        if self.fileNum != s:
            self.fileNum = s
            self.refreshFileName()

    def onCheckBoxChanged(self, state):
        checkbox = self.sender()
        self.save_type[checkbox.text()] = checkbox.isChecked()

    def collectBtnClick(self):
        if not self.collecting:
            self.collect_start()
        else:
            self.collect_stop()

    def collect_start(self):
        try:
            # get the directory path of the file
            dir_path = os.path.dirname(self.fileName)

            # check if directory exists
            if not os.path.exists(dir_path):
                # create directory if it doesn't exist
                os.makedirs(dir_path)
        except Exception as e:
            # handle the exception here
            print("An error occurred while makedirs:", e)
            return

        try:
            # open dat file
            if self.save_type["dat"]:
                self.f_byte = open(self.fileName + ".dat", "wb+")

            # open csv file
            if self.save_type["csv"]:
                self.f_csv = open(self.fileName + ".csv", "w+")
                self.f_csv.write("timestamp_low,bfee_count,noise,Nrx,Ntx,rssi_a,rssi_b,rssi_c,agc,antenna_sel,"
                                 "perm,b_len,fake_rate_n_flags,calc_len,csi\n")

            # open event file
            self.f_event = open(self.fileName + "_event.csv", "w+")
            self.f_event.write("sysTime,csiTimeStamp,pos,event\n")
        except Exception as e:
            # handle the exception here
            print("An error occurred while open file:", e)

            # close opened file
            if self.f_byte and not self.f_byte.closed:
                self.f_byte.close()
            if self.f_csv and not self.f_csv.closed:
                self.f_csv.close()
            if self.f_event and not self.f_event.closed:
                self.f_event.close()
            return

        self.collecting = True
        self.pos = 0
        self.amplitudes = []
        self.collectBtn.setText("stop collection")

        # Disable all check boxes
        for i in range(self.checkboxList.count()):
            widget = self.checkboxList.itemAt(i).widget()
            if isinstance(widget, QCheckBox):
                widget.setEnabled(False)
        # Disable edit and spinbox
        self.edit.setDisabled(True)
        self.spinbox.lineEdit().setDisabled(True)

        self.signal.emit("collect", "start")
        logText = "Start collecting data into (" + self.fileName + ") ..."
        logText += "\nClick [0-9/a-z/A-Z] on the keyboard to log the events"
        logText += "\nClick [↑↓] on the keyboard to switch the number of filename for data saving."
        logText += "\nClick [←→] on the keyboard to stop/start collecting."
        self.addLog(logText)

    def collect_stop(self, oldFileName=None):
        # Save as a local variable
        fileName = oldFileName if oldFileName else self.fileName
        amplitudes = self.amplitudes

        self.collecting = False
        self.collectBtn.setText("start collection")
        # Undisable all check boxes
        for i in range(self.checkboxList.count()):
            widget = self.checkboxList.itemAt(i).widget()
            if isinstance(widget, QCheckBox):
                widget.setEnabled(True)
        # Undisable edit and spinbox
        self.edit.setDisabled(False)
        self.spinbox.lineEdit().setDisabled(False)

        # Close the dat file
        if self.save_type["dat"]:
            self.f_byte.close()
        # Close the csv file
        if self.save_type["csv"]:
            self.f_csv.close()
        # Close event file
        self.f_event.close()

        amplitudes = np.array(amplitudes)
        # Save the npy file
        if self.save_type["npy"]:
            np.save(fileName + ".npy", amplitudes)
        # Save mat file
        if self.save_type["mat"]:
            savemat(fileName + ".mat", {'data': amplitudes})

        self.signal.emit("collect", "stop")
        self.addLog("collection stopped")

        # File number automatically +1 unless oldFileName is present (from filename modification event)
        if not oldFileName:
            self.spinbox.setValue(self.fileNum + 1)

    def keyBoardEvent(self, event):
        """
        Triggered by the keyPressEvent event of the main window
        :param event: QEvent
        :return:None
        """
        # Arrow key operation
        if event.key() == Qt.Key_Up:
            self.spinbox.setValue(self.fileNum + 1)
        if event.key() == Qt.Key_Down:
            self.spinbox.setValue(self.fileNum - 1)
        if event.key() == Qt.Key_Left and self.collecting:
            self.collect_stop()
        if event.key() == Qt.Key_Right and not self.collecting:
            self.collect_start()

        # Record event
        if event.text() and self.collecting:
            self.addLog("Event [" + event.text() + "] detected, currently logging......")
            sys_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")

            # Copy shared variables to local variables to avoid changes during calculation
            bfee_list = self.parentWindow.bfee_list.copy()
            pos = self.pos
            csi_timestamp = self.csi_timestamp

            # Corrects the time and location of events when cached data is present in the bfee_list
            if len(bfee_list) > 0:
                # csi_timestamp = int.from_bytes(bfee_list[-1][1:5], byteorder='little', signed=False)
                csi_timestamp = struct.unpack_from("<I", bfee_list[-1][1:5])[0]
                pos += len(bfee_list)

            str_write = sys_time + "," + str(csi_timestamp) + "," + str(pos) + "," + event.text() + "\n"
            self.f_event.write(str_write)
            self.f_event.flush()
            self.addLog("Logged successfully！", level="success")

    def refreshFileName(self):
        oldFileName = self.fileName
        self.fileName = self.preName + str(self.fileNum)
        self.signal.emit("filename", self.fileName)

        # restart the collection if collecting
        if self.collecting:
            self.collect_stop(oldFileName=oldFileName)
            self.addLog("File name modification detected, attempting automatic switch...")
            if len(glob.glob(self.fileName + "*")) > 0:
                self.addLog("The same-named file already exists, data collection has been automatically stopped. If "
                            "you wish to continue data collection, please click on the [Start Collection] button "
                            "or click [→] on your keyboard.", level="warning")
            else:
                self.collect_start()

    # Save raw binary data in a dat file in real time
    def collectBfee(self, bfee_use):
        if self.collecting and self.save_type["dat"]:
            for bfee in bfee_use:
                self.f_byte.write(struct.pack(">H", len(bfee)))
                self.f_byte.write(bfee)
                self.f_byte.flush()

    # Save raw binary data in a csv file in real time
    def collectCSV(self, params, csis):
        # Record the csi timestamp of the last packet
        self.csi_timestamp = params[-1][0]
        if self.collecting and self.save_type["csv"]:
            # time1 = time.perf_counter()
            str_write = ""
            for i in range(len(params)):
                param = ",".join(['"' + str(x) + '"' if isinstance(x, list) else str(x) for x in params[i]])
                csi = ",".join(["%d%+dj" % (x.real, x.imag) for x in csis[i].reshape(-1)])  # This step takes a long time (about 10 milliseconds each time).
                str_write += param + "," + csi + "\n"
            # time2 = time.perf_counter()
            self.f_csv.write(str_write)
            self.f_csv.flush()
            # time3 = time.perf_counter()
            # self.addLog("%.5f %.5f" % (time2 - time1, time3 - time2))

    # The csi amplitude data analyzed is stored as npy or mat at the end of collection
    def collectAmplitude(self, amplitude):
        # Record the current sampling location (number of packets)
        if self.collecting:
            self.pos += len(amplitude)

        # Recording amplitude
        if self.collecting and (self.save_type["npy"] or self.save_type["mat"]):
            self.amplitudes += [x for x in amplitude]

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
