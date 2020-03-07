#!/usr/bin/env python3

import sys
import socket


IP = "localhost"
PORT = 7777


def main(args):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((IP, PORT))

    while True:
        data, addr = sock.recvfrom(1024)
        print("received message: " + str(data.decode()))


if __name__ == "__main__":
    main(sys.argv)
