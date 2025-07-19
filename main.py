import os
import platform
import sys
import time

import numpy as np
import onnxruntime
from PyQt5.QtWidgets import QApplication
from scipy import signal

from MainWindow import MainWindow
from wifilib_numba_v3 import get_cv


class Config:
    def __init__(self):
        # Parameter configuration start

        # Location of the model file (Comment this line if dynamic loading is used)
        self.modelPath = "model/model.onnx"

        self.fs = 1000  # Sampling frequency
        self.cutoff = 100  # Cut-off frequency
        self.order = 8  # The order of the filter

        self.step = 1  # The minimum number of csi packages to be removed at a time
        self.Ncv = 20  # The N value of the coefficient of variation

        self.csi_window_time = 8  # Window width (seconds)

        self.Ntx = 1  # The number of transmitting antennas
        self.Nrx = 3  # The number of receiving antennas

        # The save type of the capture file
        # * the data flow is dat->csv->npy/mat
        # * for example, playing back npy files will not pick up dat.
        self.collect_types = [
            "dat",  # Raw binary data
            # "csv",  # All data parsed from binary files (read/write slowly, not recommended)
            "npy",  # The processed amplitude data can be used in numpy
            "mat",  # The processed amplitude data can be used in matlab
        ]
        # Parameter configuration end

        # Do not modify the following contents

        # Window width (Number of packets)
        self.csi_window_size = self.csi_window_time * self.fs

        # Initial parameters of Butterworth low-pass filtering
        self.ba = list(signal.butter(self.order, 2 * self.cutoff / self.fs, "lowpass"))

        # Automatic loading model
        if hasattr(self, "modelPath") and self.modelPath:
            self.ort_session = onnxruntime.InferenceSession(self.modelPath, providers=['CPUExecutionProvider'])

    def detection(self, csi, addLog):
        """
        :param csi: csi amplitude(ndarray,shape=(csi_window_size, Ntx, Nrx, 30))
        :param addLog: add log into analyzePanel
        :return: None
        """
        # preprocess
        time1 = time.perf_counter()
        xx = csi.reshape(csi.shape[0], -1)
        b, a = self.ba
        xx = signal.filtfilt(b, a, xx, axis=0)  # Bartworth filter
        time2 = time.perf_counter()
        xx = get_cv(xx, n=20)  # Coefficient of variation for n numbers
        time3 = time.perf_counter()
        xx = (xx - xx.mean()) / xx.std()
        inputs = xx.reshape(1, 1, 400, 90).astype(np.float32)

        # Predic
        isFall = self.ort_session.run(
            None,
            {self.ort_session.get_inputs()[0].name: inputs},
        )[0].argmax()

        # output
        time4 = time.perf_counter()
        if isFall:
            addLog("Fall detected", level="warning")
            duration = 0.05
            freq = 440
            plat = platform.system().lower()
            if plat == 'linux':
                os.system('play --no-show-progress --null --channels 1 synth %s sine %f &' % (duration, freq))
        else:
            addLog("No fall detected")
        time5 = time.perf_counter()
        print("fall" if isFall else "noFall",
              "%.4f %.4f %.4f %.4f" % (time2 - time1, time3 - time2, time4 - time3, time5 - time4))

if __name__=="__main__":
    app = QApplication(sys.argv)
    config = Config()
    mainWindow = MainWindow(config)
    mainWindow.show()
    sys.exit(app.exec_())
