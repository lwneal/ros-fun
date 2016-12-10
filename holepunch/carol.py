import socket
import time

PORT = 33133
alice_addr = ('',0)
bob_addr = ('',0)

while True:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(('0.0.0.0', PORT))
    msg, addr = s.recvfrom(2048)
    client_ip, client_port = addr
    print("Recv from {}:{}: {}".format(client_ip, client_port, msg))
    if 'this is Alice' in msg:
        client_name = 'Alice'
        alice_addr = addr
    elif 'this is Bob' in msg:
        client_name = 'Bob'
        bob_addr = addr
    else:
        print("Error: unknown client")
        continue

    if client_name is 'Alice':
        payload = '{} {}'.format(*bob_addr)
    else:
        payload = '{} {}'.format(*alice_addr)

    other_client_name = 'Bob' if client_name is 'Alice' else 'Alice'

    print("Responding to {} at {}:{}".format(client_name, client_ip, client_port))

    s.sendto(payload, (client_ip, client_port))
    print("Sent {} bytes to ip {} port {}".format(len(payload), client_ip, client_port))
