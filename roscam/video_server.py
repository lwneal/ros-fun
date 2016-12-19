import socket
import struct
import sys

def decode_jpg(data):
    from PIL import Image
    from cStringIO import StringIO
    fp = StringIO(data)
    return Image.open(fp)

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

s.bind(('0.0.0.0', 1234))

s.listen(1)

conn, addr = s.accept()

sys.stderr.write("Recv connection from {} {}\n".format(conn, addr))

idx = 0

while True:
    packet_type = ord(conn.recv(1))
    if packet_type is None:
        break
    packet_len = struct.unpack('!l', conn.recv(4))[0]
    sys.stderr.write("Got packet type {} length {}... ".format(packet_type, packet_len))

    packet_data = ""
    while len(packet_data) < packet_len:
        packet_data = packet_data + conn.recv(packet_len - len(packet_data))
    sys.stderr.write("recv {} bytes\n".format(len(packet_data)))

    frame = decode_jpg(packet_data)

    # TODO: Display frame somehow
    frame.save('static/visual.jpg')

    # TODO: Run frame through computer vision system
    #sys.stdout.write(packet_data)

sys.stderr.write("Connection closed\n")
conn.close()
