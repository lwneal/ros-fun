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
payload = "Hi Carol, this is Alice. Can I talk to Bob?"
s.sendto(payload, (CAROL_IP, LISTEN_PORT))
print("Sent {} bytes".format(len(payload)))

# Now Alice waits for a response from Carol
print("Now listening on port {}".format(LISTEN_PORT))
response, addr = s.recvfrom(2048)
print('Recv from {}: {}'.format(addr, response))

bob_ip, bob_port = response.split()
bob_port = int(bob_port)

while True:
    print("Sending packet to {}:{}".format(bob_ip, bob_port))
    payload = "Bob, our love trancends network address translation!" 
    s.sendto(payload, (bob_ip, bob_port))
    data, addr = s.recvfrom(2048)
    print addr, data
