"""Temporary loopback CONNECT relay for this run's Linux API connectivity.

It accepts only api.openai.com:443 and relays opaque TLS bytes. Certificate
verification remains in the Linux HTTPS client. No headers or payloads are
logged. Stop this process and its SSH reverse forward when qualification ends.
"""
import select
import socket
import socketserver


class Connect(socketserver.BaseRequestHandler):
    def handle(self):
        self.request.settimeout(15)
        header = b""
        while b"\r\n\r\n" not in header and len(header) < 8192:
            chunk = self.request.recv(1024)
            if not chunk:
                return
            header += chunk
        first = header.split(b"\r\n", 1)[0].split()
        if len(first) != 3 or first[:2] != [b"CONNECT", b"api.openai.com:443"]:
            self.request.sendall(b"HTTP/1.1 403 Forbidden\r\nContent-Length: 0\r\n\r\n")
            return
        with socket.create_connection(("api.openai.com", 443), timeout=15) as upstream:
            self.request.sendall(b"HTTP/1.1 200 Connection Established\r\n\r\n")
            rest = header.split(b"\r\n\r\n", 1)[1]
            if rest:
                upstream.sendall(rest)
            pair = [self.request, upstream]
            while True:
                readable, _, _ = select.select(pair, [], [], 60)
                if not readable:
                    return
                for source in readable:
                    data = source.recv(65536)
                    if not data:
                        return
                    (upstream if source is self.request else self.request).sendall(data)


class Server(socketserver.ThreadingTCPServer):
    daemon_threads = True


if __name__ == "__main__":
    with Server(("127.0.0.1", 0), Connect) as server:
        print(server.server_address[1], flush=True)
        server.serve_forever()
