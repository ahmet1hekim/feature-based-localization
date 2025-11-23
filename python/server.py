import socket
import numpy as np
import cv2
import struct


def recvall(sock, count):
    buf = b""
    while len(buf) < count:
        newbuf = sock.recv(count - len(buf))
        if not newbuf:
            return None
        buf += newbuf
    return buf


def recv_mat(sock):
    meta_size = 12  # 3 ints: rows, cols, type
    meta = recvall(sock, meta_size)
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

    data = recvall(sock, data_size)
    if data is None:
        return None

    mat = np.frombuffer(data, dtype=dtype)
    if channels > 1:
        mat = mat.reshape((rows, cols, channels))
    else:
        mat = mat.reshape((rows, cols))

    return mat


def main():
    TCP_IP = "0.0.0.0"  # listen on all interfaces
    TCP_PORT = 12345

    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.bind((TCP_IP, TCP_PORT))
    server_sock.listen(1)

    print("Waiting for connection...")
    conn, addr = server_sock.accept()
    print("Connected by:", addr)

    cv2.namedWindow("Received", cv2.WINDOW_AUTOSIZE)

    try:
        while True:
            mat = recv_mat(conn)
            if mat is None:
                print("Connection closed or failed to receive")
                break

            cv2.imshow("Received", mat)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC to quit
                print("ESC pressed, exiting...")
                break

    finally:
        conn.close()
        server_sock.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
