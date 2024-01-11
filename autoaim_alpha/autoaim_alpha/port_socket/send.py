
import socket
s = socket.socket()
s.bind(('127.0.0.1',9006))
s.listen(5)
print("listening...")

while 1:
    sock,addr = s.accept()
    print(sock,addr)
    while 1:
        text = sock.recv(1024)
        if len(text.strip()) == 0:
            print("none from client")
        else:
            print("message from client:{}".format(text.decode()))
            content = input("please write to send back to client")
            sock.send(content.encode())

    sock.close()

