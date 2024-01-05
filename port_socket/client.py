import socket
s1 = socket.socket()
s1.connect(('127.0.0.1', 9006))
while 1:
    send_data = input("write to send to server:")
    s1.send(send_data.encode())
    text = s1.recv(1024).decode()
    print("message from server:{}".format(text))
    print("------------------------------")
    
    