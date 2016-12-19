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

t = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
t.bind(('0.0.0.0', 1235))
t.listen(1)
sys.stderr.write("Waiting for connection: {}".format(t))
conn2, addr2 = t.accept()
sys.stderr.write("Recv connection from {} {}\n".format(conn2, addr2))

idx = 0

while True:
    packet_type_bytes = conn.recv(1)
    packet_type = ord(packet_type_bytes)
    conn2.send(packet_type_bytes)

    packet_len_bytes = conn.recv(4)
    conn2.send(packet_len_bytes)
    packet_len = struct.unpack('!l', packet_len_bytes)[0]
    sys.stderr.write("Got packet type {} length {}... ".format(packet_type, packet_len))

    packet_data = ""
    while len(packet_data) < packet_len:
        packet_data_bytes = conn.recv(packet_len - len(packet_data))
        conn2.send(packet_data_bytes)
        packet_data = packet_data + packet_data_bytes
    sys.stderr.write("recv {} bytes\n".format(len(packet_data)))

    frame = decode_jpg(packet_data)
    # TODO: Display frame somehow
    frame.save('static/visual.jpg')
    # TODO: Run frame through computer vision system
    #sys.stdout.write(packet_data)

sys.stderr.write("Connection closed\n")
conn.close()
conn2.close()
