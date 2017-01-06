import sys
import socket
import util


ADDR = (('localhost', 1237))


def resnet(jpg_data):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect(ADDR)
    util.write_packet(s, jpg_data)
    response_type, response_data = util.read_packet(s)
    return response_data


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: {} input.jpg > output.jpg".format(sys.argv[0]))
        exit()
    jpg_data = open(sys.argv[1]).read()
    sys.stdout.write(resnet(jpg_data))
