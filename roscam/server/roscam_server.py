"""
The server listens on port 33133.
Each time a client connects, the server spawns a subprocess.
"""
import SocketServer

class TCPHandler(SocketServer.BaseRequestHandler):
    def handle(self):
        self.data = self.request.recv(1024)
        self.request.sendall('ok I heard you')
    
if __name__ == '__main__':
    server = SocketServer.TCPServer(('localhost', 33133), TCPHandler)
    server.serve_forever()
