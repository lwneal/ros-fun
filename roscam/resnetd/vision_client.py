import sys
import socket
import util


def resnet(addr, jpg_data):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect(addr)
    util.write_packet(s, jpg_data)
    response_type, response_data = util.read_packet(s)
    return response_data


if __name__ == '__main__':
    addr = ('127.0.0.1', 1237)
    if len(sys.argv) < 2:
        print("Usage: {} input.jpg [server] > output.jpg".format(sys.argv[0]))
        print("server: defaults to localhost")
        exit()
    jpg_data = open(sys.argv[1]).read()
    if len(sys.argv) > 2:
        addr = (sys.argv[2], 1237)
    sys.stdout.write(resnet(addr, jpg_data))
