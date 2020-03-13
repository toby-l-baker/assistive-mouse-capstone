#!/usr/bin/env python3

import sys
import socket


def main(args):
    if len(args) != 3:
        print("Usage: address port")
        exit()
    
    address = args[1]
    port = int(args[2])

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((address, port))

    while True:
        data, addr = sock.recvfrom(1024)
    
        keypoints = data.decode().strip(';').split(';')

        for keypoint in keypoints:
            x, y, z = [float(i) for i in keypoint.strip(',').split(',')]
            print(str((x, y, z)))

        print('--------------')


if __name__ == "__main__":
    main(sys.argv)
