import json
import re
import sys
import time
from datetime import datetime
from threading import Thread
from types import SimpleNamespace

import numpy as np
from PyQt5.QtCore import Qt, QTimer, QSettings, QPoint, QThread, pyqtSignal
from PyQt5.QtWidgets import QApplication, QMainWindow, QDockWidget, QTextEdit, QTabWidget, QTabBar, QWidget, \
    QVBoxLayout, QLabel, QFrame, QSlider, QGridLayout, QHBoxLayout, QFileDialog, QMessageBox, QProgressBar, \
    QProgressDialog
from numba.typed import List
from scipy import signal
import pyqtgraph as pg

from Playback import Playback
from SocketClient import SocketClient
from ui import SignalPanel, CollectPanel, AnalysePanel
from wifilib_numba_v3 import get_cv, analyse_bf, get_scale_csi, fillna


class CompileThread(QThread):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parentWindow = parent

    def run(self):
        # Processing a csi signal in a child thread triggers numba precompilation
        # the first compilation takes a long time and requires patience
        time1 = time.perf_counter()
        bfee_use = [
            b'\xbb5\xdf%\xbaw`\x00\x00\x03\x03$&"\x81!!(\x02\x13\x018\xea\xaf\xf07\xc1\x18\x81\x88\xbe6\x81\x8878\xd0\xd0~\x06\x13<\x82D\xd0\x04\x08\x011v\x8aA\xbc\xc7\x06Fs6t\xca\xe7U\x98 4\xf0\x83\xb3Q\xfc\xf9eH\x1c\x98\xd3\x11\xd2\x8d>\xc4\x04!\xf1>\xbc\x9d\x02\xaf \x04"p\xed\x8f\x01\xeb\xee\xa9\x9c\x88\x87pen\x95q\t&\n\x7fs\x08\xb0\'K[i\\8xS\x83\xbb`K0\x11\xd8\xd3k\x80\xbc\xd7\xb9I\xbf\xe2\x01\xfc;\xdd\xe6z\xe1\x08\x9e}_\x04\xe5\xb1\xd4>\xda\n\n\xe2\xe8\xf2B\xd8\x088\xe3\xe0\xfb%\x98w-\xa7\x01\xb6\'\x10_\xe7W\xeaF8\xd1~\xee\x1fAAn\xfe\n\xed6?7\xfa\x01\xd3:\x82\x05v\xf1~\x062\xa6\x1bff\x89\xe3\xab\xc3\x17\x94\xee!\x10\xc4y\xf9\x17\xe0a\x7f\x01\xf4\x9b\xbb\x9e\x1c\xde\xe0\xe3\xaf\xe1\xbf\xce\x9b\xc0\x0f\x8a\x0e\x0c(\xe7\xdex\xdb\xef\x84\x15\x00\x11\xfe\xfca\x0cw\x04\xdc$\\]\'\xeb\xbb\x9a+|\xf0\x8b\x0c\x04L\xab\x98\x83\xbeH\xc0\n{Z\x80\x95\x9da\xc3>\x84\x80\xa0;\x06\x1d\xd7Q\xf0H\xd4\xe6\x11\xaf\xf4\x14\x14\xee\x18\x06\x02\xee?\xef\x10v\n/)\x9e\xef\xa8\xed/\xb9\x18_(\xb8\xdf/\xd2\xc7.\x90\xb7\xc0\xac~\x88\xb1~\r\x07\xf8~?\xbb\x80\xd1\x01zk\xd4\xc5M\xe1=\xa8\xd9\x89@\xca\xdf\xdf\xc7\x05v&\xc0\x8c\xe2_l:\xfe \xde\xfd\xb4\x81\x8e\xde\xed\xdd\x9f"\x81o\x99\x8e[\xdc\xe7\x7f\xef\xe3\xa6\x87x\xf6\xe3o\xfb\x0b\x08\xc0\x03\xd9\xdcN3\xd7c\xf7\xfa\xec\xc3\xdb\xf7\xb2\xcf+\x14\xa0\x9eJH\x17\xbe\x9a\x1d:7f\xdd{\x80V\xff\x9e \x1e\xf7d@\xbe\x11\xe2\xef\xc9\xc3%\xdb\xcf\x0e\xb9\t\x06\x01\xe4\xb8Wc\x11v\tpg\xbe\xee\xd0\xbe\x95 \xc6\xa8(\xc8\x86\xbcZ\x84pP\x05\xfd5\xbb\x87\xf7\xaa\x83\xb4\x88\xc9\x815\xda\xad\xe2o\xa5@\xe0\xdf\x01P\xce=\x07\xccWj\x08\xb6\x11O\x03\xad\x0b\xe5Q>\xd0\x10\xf3~\xba\xafOc\x03\x80\x0ev\x89]\xe4&\x05\xea\x0f\x8b\x1a\xf8W\xf9\x87\x9a\x15\xfe|\xac\x1f\xe8J\xfb\xfcK\x8fT\xc0\xc0\x07\xd3O\xbc\x84\xf0\x0f\x00']
        params, csis = analyse_bf(List(bfee_use))
        csi_complex = get_scale_csi(csis, params)
        csi_abs = np.abs(csi_complex)
        csi_abs = fillna(csi_abs)
        time2 = time.perf_counter()
        print("The initialization was complete, taking %.3f seconds" % (time2 - time1))


class MainWindow(QMainWindow):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # self.load_config()
        self.csi_cache = np.ones((self.config.csi_window_size, self.config.Ntx, self.config.Nrx, 30))  # csi缓存
        self.total_length = 0

        # self.bfee_list = List()
        self.bfee_list = []

        # Sets the title and size of the main window
        self.setWindowTitle("PyQT CsiTool")
        self.setGeometry(100, 100, 800, 800)

        self.initUI()
        self.generate_image()
        self.updateGraph()
        self.setEnabled(False)
        self.show()
        self.compile()


        # Periodically convert cached binary data to csi
        timer = QTimer(self)
        timer.setTimerType(Qt.PreciseTimer)
        timer.timeout.connect(self.listen_bfee)
        timer.start(100)

    def compile(self):
        # display the "Initializing" window
        self.progress_dialog = QProgressDialog("Initializing...", None, 0, 0, self)
        self.progress_dialog.setWindowTitle("Info")
        self.progress_dialog.setFixedSize(300, 100)
        progress_bar = QProgressBar()
        progress_bar.setRange(0, 0)
        progress_bar.setAlignment(Qt.AlignCenter)
        self.progress_dialog.setBar(progress_bar)
        self.progress_dialog.show()

        # Precompile functions using numba in child threads
        self.compile_thread = CompileThread(self)
        self.compile_thread.finished.connect(self.compiled)
        self.compile_thread.start()

    # Close the load window after compiling
    def compiled(self):
        self.setEnabled(True)
        self.progress_dialog.hide()


    def initUI(self):
        # Sets the container for pyqtgraph
        self.frame = QFrame(self)  # Create a parent container
        self.frame.setFrameShape(QFrame.Panel)
        self.frame.setFrameShadow(QFrame.Plain)
        self.frame.setLineWidth(2)
        self.frame.setStyleSheet("background-color:#888888;")

        # Set slider to change the cutoff frequency of the low-pass filter
        self.slider = QSlider(Qt.Horizontal, self)
        self.slider.setGeometry(50, 50, 200, 30)
        self.slider.setMinimum(1)
        self.slider.setMaximum(self.config.fs // 2 - 1)
        self.slider.setValue(self.config.cutoff)
        self.slider.setSingleStep(1)
        self.slider.valueChanged.connect(self.updateCutoff)

        self.sliderText = QLabel(self)
        self.sliderText.setText("cutoff of filter: " + str(self.config.cutoff) + "Hz")

        self.gridLayout = QGridLayout(self)
        self.gridLayout.addWidget(self.frame, 0, 0)
        self.gridLayout.addWidget(self.sliderText, 1, 0)
        self.gridLayout.addWidget(self.slider, 2, 0)

        widget = QWidget()
        widget.setLayout(self.gridLayout)
        self.setCentralWidget(widget)

        self.addDockWidget1()
        self.addDockWidget2()
        self.addDockWidget3()
        self.setDockOptions(self.dockOptions() | QMainWindow.AllowNestedDocks | QMainWindow.VerticalTabs)

    def addDockWidget1(self):
        # Create signal Panel
        dockWidget = QDockWidget("Signal Panel", self)
        dockWidget.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea | Qt.BottomDockWidgetArea)

        self.signalPanel = SignalPanel(self.bfee_list, self.config, parent=self)
        self.signalPanel.signal.connect(self.signalPanelEvent)
        dockWidget.setWidget(self.signalPanel)

        dockWidget.setMinimumWidth(200)
        self.addDockWidget(Qt.BottomDockWidgetArea, dockWidget)

    def signalPanelEvent(self, event):
        if event == "monitor start":
            pass
        elif event == "monitor stop":
            pass
        elif event == "play start":
            pass
        elif event == "play stop":
            pass

    def addDockWidget2(self):
        # Create Collection Panel
        dockWidget = QDockWidget("Collection Panel", self)
        dockWidget.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea | Qt.BottomDockWidgetArea)

        self.collectPanel = CollectPanel(self.config, self)
        self.collectPanel.signal.connect(self.collectPanelEvent)
        dockWidget.setWidget(self.collectPanel)

        # dockWidget.setMinimumWidth(300)
        # dockWidget.setMinimumHeight(200)
        self.addDockWidget(Qt.BottomDockWidgetArea, dockWidget)

    def collectPanelEvent(self, event, param):
        if event == "collect":
            if param == "start":
                self.region.setRegion([self.config.csi_window_time, self.config.csi_window_time])
            elif param == "stop":
                pass
        elif event == "filename":
            self.saveFileName = param
            print(self.saveFileName)

    def keyPressEvent(self, event):
        # Sends the keyboard events to the collectPanel
        if self.collectPanel:
            self.collectPanel.keyBoardEvent(event)
        return super().keyPressEvent(event)

    def addDockWidget3(self):
        # Create Analysis Panel
        dockWidget = QDockWidget("Analysis Panel", self)
        dockWidget.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea | Qt.BottomDockWidgetArea)

        self.analysePanel = AnalysePanel(self.csi_cache, self.config, self)
        self.analysePanel.signal.connect(self.analysePanelEvent)
        dockWidget.setWidget(self.analysePanel)

        self.addDockWidget(Qt.BottomDockWidgetArea, dockWidget)

    def analysePanelEvent(self, event):
        pass

    def generate_image(self):
        win = pg.GraphicsLayoutWidget(self.frame)  # Display Graphics on the frame
        verticalLayout = QHBoxLayout(self.frame)  # Add graph to the parent container
        verticalLayout.addWidget(win)

        p1 = win.addPlot(title="Waveform Graph of Amplitude")
        # p1.buttonsHidden=True # Hide the auto button in the lower left corner
        # p1.setYRange(0, 30) # Set the Y-axis range
        p1.showGrid(x=True, y=True)
        p1.setLabel(axis="left", text="Amplitude")
        # p1.setLabel(axis="bottom", text="Time (s)")
        p1.setMouseEnabled(y=False)
        p1.addLegend()

        # Shadow range
        self.region = pg.LinearRegionItem([self.config.csi_window_time, self.config.csi_window_time], movable=False)
        p1.addItem(self.region)

        win.nextRow()
        p2 = win.addPlot(title="Waveform Graph of GCV")
        p2.showGrid(x=True, y=True)
        p2.setLabel(axis="left", text="GCV")
        p2.setLabel(axis="bottom", text="Time (s)")
        p2.setMouseEnabled(y=False)
        p2.addLegend()

        # Bind the X axis of p1 and p2 plotItem to realize synchronous scrolling
        p2.getViewBox().setXLink(p1.getViewBox())

        # pg.setConfigOptions(antialias=True)  # Anti-aliasing, smoother graphics but higher performance overhead

        self.curve = {"origin": [
            p1.plot(pen="#663333dd"),
            p1.plot(pen='#336633dd'),
            p1.plot(pen='#333366dd'),
            # p1.plot(pen='#FFFFFF')
        ], "filtered": [
            p1.plot(pen="r", name="y0"),
            p1.plot(pen='g', name="y1"),
            p1.plot(pen='b', name="y2")
        ], "cv": [
            p2.plot(pen="r", name="y0"),
            p2.plot(pen='g', name="y1"),
            p2.plot(pen='b', name="y2")
        ]}

    def listen_bfee(self):
        time0 = time.perf_counter()
        length_use = len(self.bfee_list) // self.config.step * self.config.step
        if length_use > 0:
            # get length_use package
            bfee_use = self.bfee_list[:length_use]
            self.bfee_list[:] = self.bfee_list[length_use:]
            self.collectPanel.collectBfee(bfee_use)  # Send the original binary data to the collector and save it in dat file in real time

            csi_abs = self.analyse_bfee_use(bfee_use)
            self.addCache(csi_abs)
            self.updateRegion(len(csi_abs))
            time1 = time.perf_counter()

            self.updateGraph()
            time2 = time.perf_counter()

            # Displays the number and format of received packets in the log
            self.signalPanel.addLog("recv:" + str(csi_abs.shape))
            # self.signalPanel.log.append(time.strftime("%H:%M:%S.")+"\n"+str(csi_abs.shape))
            # Shows how long it takes to analyze the amplitude and how long it takes to update the view
            print(self.csi_cache.shape, csi_abs.shape, "%.5f %.5f" % (time1 - time0, time2 - time1))

    # Parse the byte
    def analyse_bfee_use(self, bfee_use):
        params, csis = analyse_bf(List(bfee_use))
        self.collectPanel.collectCSV(params, csis)  # The original csi data analyzed is sent to the collector and saved in real time as csv files
        csi_complex = get_scale_csi(csis, params)
        # csi_abs = np.abs(csi_complex)
        csi_abs = np.abs([x[:self.config.Ntx, :self.config.Nrx, :]
                          for x in csi_complex if x.shape[0] >= self.config.Ntx and x.shape[1] >= self.config.Nrx])

        return csi_abs

    # Write the amplitude to the cache
    def addCache(self, csi_abs):
        self.collectPanel.collectAmplitude(csi_abs)  # The csi amplitude data analyzed is sent to the collector and saved in npy file or mat file at the end of collection
        csi_abs = fillna(csi_abs)  # Fill null value
        len_package = len(csi_abs)
        self.total_length += len_package
        self.csi_cache[:-len_package] = self.csi_cache[len_package:]
        self.csi_cache[-len_package:] = csi_abs[:self.config.csi_window_size]

    # Update shaded area
    def updateRegion(self, len_package):
        # Moving shadow
        region_area = list(self.region.getRegion())
        if self.collectPanel.collecting:  # When sampling, move the front boundary
            region_area[0] -= len_package / self.config.fs
        else:  # When not sampling, move the double boundary
            region_area[0] -= len_package / self.config.fs
            region_area[1] -= len_package / self.config.fs
        if region_area[0] < 0:
            region_area[0] = 0
        if region_area[1] < region_area[0]:
            region_area[1] = region_area[0]
        self.region.setRegion(region_area)

    # Update the cache data to the graphics window
    def updateGraph(self):
        csi_cache = self.csi_cache
        Ncv = self.config.Ncv

        # The X-axis is fixed at 0 to 8
        x = np.arange(0, len(self.csi_cache)) / self.config.fs
        # The X-axis is moving over time
        # x = (np.arange(0*Ncv, len(csi_cache))+self.total_length) / self.config.fs

        self.curve["origin"][0].setData(x, csi_cache[:, 0, 0, 0])
        self.curve["origin"][1].setData(x, csi_cache[:, 0, 1, 0])
        self.curve["origin"][2].setData(x, csi_cache[:, 0, 2, 0])

        csi_filter = signal.filtfilt(self.config.ba[0], self.config.ba[1], csi_cache[:, 0:1, :, 0:1], axis=0)
        csi_filter = np.around(csi_filter, 1)  # Avoid Y axis flickering

        self.curve["filtered"][0].setData(x, csi_filter[:, 0, 0, 0])
        self.curve["filtered"][1].setData(x, csi_filter[:, 0, 1, 0])
        self.curve["filtered"][2].setData(x, csi_filter[:, 0, 2, 0])

        csi_cv = get_cv(np.roll(csi_filter, self.total_length % Ncv, axis=0), n=Ncv)
        # print(-(self.total_length % Ncv))
        self.curve["cv"][0].setData(x[Ncv::Ncv], csi_cv[1::, 0, 0, 0])
        self.curve["cv"][1].setData(x[Ncv::Ncv], csi_cv[1::, 0, 1, 0])
        self.curve["cv"][2].setData(x[Ncv::Ncv], csi_cv[1::, 0, 2, 0])

    def updateCutoff(self, num):
        self.config.cutoff = num
        self.sliderText.setText("cutoff of filter: " + str(self.config.cutoff) + "Hz")
        self.config.ba[:] = signal.butter(self.config.order, 2 * self.config.cutoff / self.config.fs, "lowpass")
        self.updateGraph()
