import socket
import torch
import torch.nn as nn
import pickle

host = '127.0.0.1'
port = 9999
buffer_size = 4096


class ClientModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        return x

class ClientClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        #x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def send_tensor(file, socket):
    buffer = 4096
    for buffer in file:
        socket.send(buffer)
    file.close()
    socket.send(b"Done")
    return


def receive_tensor(filename, socket):
    recvd = socket.recv(buffer_size)
    with open(filename, 'wb') as file:
        while (not(str(recvd).__contains__('Done'))):
            file.write(recvd)
            recvd = socket.recv(buffer_size)
        file.write(recvd)
    return

def create_socket_and_connect(host, port):
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        print(f"[+] Connecting to {host}:{port}")
        s.connect((host, port))
        print("[+] Connected.")
        return s
    except socket.error as err:
        print(f"Socket creation failed with error {err}")


server = create_socket_and_connect(host, port)
receive_tensor('starting_weights.pkl', server)
#print('Recieved Weights')
server_weights = pickle.load(open('starting_weights.pkl', 'rb'))
#print(f'Server Weights: {server_weights}')
client_model = ClientModel()
client_classifier = ClientClassifier()
client_model.load_state_dict(server_weights)
print(client_model.state_dict())