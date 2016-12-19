import socket
import time

# Alice and Bob are star crossed lovers who wish only to communicate directly over the Internet
# Tragically, each one is trapped behind an oppressive corporate firewall
# Carol is in the free zone outside of the walls
# She will help to reunite Alice and Bob

# A "matchmaking" server that we control
CAROL_IP = "173.255.248.229"
LISTEN_PORT = 33133

s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s.bind(('0.0.0.0', LISTEN_PORT))
payload = "Hello, this is Bob. I wish to connect to Alice"
s.sendto(payload, (CAROL_IP, LISTEN_PORT))
print("Sent {} bytes".format(len(payload)))

# Now Alice waits for a response from Carol
print("Now listening on port {}".format(LISTEN_PORT))
response, addr = s.recvfrom(2048)
print('Recv from {}: {}'.format(addr, response))

alice_ip, alice_port = response.split()
alice_port = int(alice_port)

while True:
    print("Sending packet to {}:{}".format(alice_ip, alice_port))
    payload = "Alice, my love- it is I, Bob! No firewall can keep us apart."
    s.sendto(payload, (alice_ip, alice_port))
    data, addr = s.recvfrom(2048)
    print addr, data
