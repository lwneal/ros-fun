"""
The server listens on port 33133.
It listens for one client.
It gets data from the client and sends it to block storage
"""
import SocketServer

class TCPHandler(SocketServer.BaseRequestHandler):
    def handle(self):
        print("server got connection...")
        self.data = self.request.recv(2048)
        self.request.sendall('ok I heard you')
    
if __name__ == '__main__':
    server = SocketServer.TCPServer(('localhost', 33133), TCPHandler)
    print("starting server")
    server.serve_forever()
