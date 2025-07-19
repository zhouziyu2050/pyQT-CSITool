import socket
from struct import unpack
from threading import Thread
import os

from PyQt5.QtCore import QThread


class SocketClient(QThread):
    """
    Data is received in the QThread through the socket, and stored in the shared memory variable 'bfee_list'
    """

    def __init__(self, bfee_list, parent=None):
        super().__init__(parent)
        self.bfee_list = bfee_list
        self.connecting = True

    def run(self):
        NETLINK_CONNECTOR = 11

        # The sudo permission is required here
        self.sk = socket.socket(socket.AF_NETLINK, socket.SOCK_DGRAM, NETLINK_CONNECTOR)
        self.sk.bind((os.getpid(), -1))

        print('start connect...')
        while self.connecting:
            try:
                buff, address = self.sk.recvfrom(4096)
                length = buff[32] + buff[33] * 256
                data = buff[36:int(length) + 36]
                self.bfee_list.append(data)
            except Exception as e:
                print(e)  # recvfrom error
                self.stop()
                break
        print('connect closed!')

    def stop(self):
        self.connecting = False
        self.sk.close()

    def printbyte(self, b):
        for i in b:
            print("\\x%x" % i, end="")
        print("")
