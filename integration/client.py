#!/usr/bin/env python3

import sys
import socket


IP = "localhost"
PORT = 7777


def main(args):
    msg = 'hello world!'.encode()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.sendto(msg, (IP, PORT))


if __name__ == "__main__":
    main(sys.argv)
