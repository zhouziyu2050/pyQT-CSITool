import time
from PyQt5.QtCore import QThread

from wifilib_numba_v3 import split_bytes


class Playback(QThread):
    def __init__(self, path, bfee_list, freq=100):
        super().__init__()
        self.path = path
        self.bfee_list = bfee_list
        self.freq = freq

        self.interval = 1 / self.freq
        self.running = True


    def run(self):
        print("Start to load playback data. The playback file is:", self.path)
        self.binary_data = self.read_byte_array(self.path)  # Load file
        self.seek = 0  # Pointer set to 0
        next_time = time.perf_counter()

        # After the loop is complete, the finished signal is generated and the thread exits
        while self.running:
            self.sendData()
            next_time += self.interval
            sleep_time = max(0, next_time - time.perf_counter())
            time.sleep(sleep_time)

    def sendData(self):
        if self.seek < len(self.binary_data):
            buff = self.binary_data[self.seek]
            self.bfee_list.append(buff)
            self.seek += 1
            # print(".", end="")
        else:  # When the playback is over, emit the stop command
            self.stop()

    def stop(self):
        # After the loop is complete, the finished signal is generated and the thread exits
        self.running = False

    def read_byte_array(self, path):
        with open(path, mode='rb') as f:
            buffer = f.read()
        bfee_list = split_bytes(buffer)
        return bfee_list
