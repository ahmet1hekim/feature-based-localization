import socket
import struct
import threading

import numpy as np


class Receiver:
    def __init__(self, IP, PORT):
        self.TCP_IP = IP
        self.TCP_PORT = PORT

        self.mat = None
        self.stop_flag = False
        self.mat_lock = threading.Lock()

        # Create client socket and connect to server
        self.conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        print(f"Connecting to server at {self.TCP_IP}:{self.TCP_PORT}...")
        self.conn.connect((self.TCP_IP, self.TCP_PORT))
        print("Connected to server.")

        # Thread for receiving data
        self.thread = threading.Thread(target=self.loop, daemon=True)

    def start(self):
        self.thread.start()

    def stop(self):
        self.stop_flag = True
        self.thread.join()
        self.conn.close()

    # -------------------------
    # Low-level receive helpers
    # -------------------------
    def recvall(self, count):
        buf = b""
        while len(buf) < count:
            newbuf = self.conn.recv(count - len(buf))
            if not newbuf:
                return None
            buf += newbuf
        return buf

    # -------------------------
    # Receive a single OpenCV mat
    # -------------------------
    def recv_mat(self):
        meta_size = 12  # 3 ints: rows, cols, type
        meta = self.recvall(meta_size)
        if meta is None:
            return None

        rows, cols, type_ = struct.unpack("iii", meta)

        depth = type_ & 7
        channels = 1 + (type_ >> 3)

        depth_to_dtype = {
            0: np.uint8,
            1: np.int8,
            2: np.uint16,
            3: np.int16,
            4: np.int32,
            5: np.float32,
            6: np.float64,
        }
        dtype = depth_to_dtype.get(depth, np.uint8)

        elem_size = np.dtype(dtype).itemsize * channels
        data_size = rows * cols * elem_size

        data = self.recvall(data_size)
        if data is None:
            return None

        mat = np.frombuffer(data, dtype=dtype)

        if channels > 1:
            mat = mat.reshape((rows, cols, channels))
        else:
            mat = mat.reshape((rows, cols))

        return mat

    # -------------------------
    # Thread loop
    # -------------------------
    def loop(self):
        while not self.stop_flag:
            m = self.recv_mat()
            if m is None:
                print("Server disconnected.")
                break

            # thread-safe update of latest matrix
            with self.mat_lock:
                self.mat = m

    # -------------------------
    # Thread-safe getter
    # -------------------------
    def get_mat(self):
        with self.mat_lock:
            if self.mat is None:
                return None
            return self.mat.copy()  # safe


# ---------------------------------------------------
# Example usage
# ---------------------------------------------------
# recv = Receiver("127.0.0.1", 12345)
# recv.start()
#
# while True:
#     frame = recv.get_mat()
#     if frame is not None:
#         print("Received shape:", frame.shape)
