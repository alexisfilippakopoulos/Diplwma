import socket
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision
import torchvision.transforms as transforms
import pickle
import threading
import queue

event = threading.Event()
file_lock = threading.Lock()
file_condition = threading.Condition()
serverip = 'localhost'
serverport = 9999
buffer_size = 4096
global client_socket



# Downloading Dataset
def get_dataset():
    transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))])

    # Create datasets for training & validation, download if necessary
    training_set = torchvision.datasets.FashionMNIST('./data', train=True, transform=transform, download=True)
    test_set = torchvision.datasets.FashionMNIST('./data', train=False, transform=transform, download=True)
    
    return training_set, test_set

# Creating dataloaders
def create_dataloader(training_set, test_set):
    batch_size = 4
    val_ratio = 0.2
    train_dataset, val_dataset = random_split(training_set, 
                                            [int((1 - val_ratio) * len(training_set)), 
                                            int(val_ratio * len(training_set))])
    train_dl = DataLoader(train_dataset, 
                        batch_size, 
                        shuffle=True, 
                        num_workers=3, 
                        pin_memory=True)
    valid_dl = DataLoader(val_dataset,
                        batch_size,
                        shuffle=True,
                        num_workers=3,
                        pin_memory=True)
    test_dl = DataLoader(test_set, 
                        batch_size, 
                        num_workers=3, 
                        shuffle=True, 
                        pin_memory=3)
    return train_dl, valid_dl, test_dl

def get_default_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

def to_device(data, device):
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
        
    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl: 
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)

def create_socket_and_listen(serverIp, serverPort):
    try:
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        print("Socket successfully created")
        server.bind((serverIp, serverPort))
        server.listen(5)
        listen_thread = threading.Thread(target=listen_for_connections, args=(server,))
        listen_thread.start()
        return server
    except socket.error as err:
        print(f"Socket creation failed with error {err}")

def send_data(filename, socket, data):
    pickle.dump(data, open(filename, 'wb'))
    buffer = 4096
    with open(filename, 'rb') as file:
        for buffer in file:
        #print(buffer)
            socket.send(buffer)
    socket.send(b"Done")
    return

def receive_data(filename, socket):
    recvd = socket.recv(buffer_size)
    with open(filename, 'wb') as file:
        while (not(str(recvd).__contains__('Done'))):
            file.write(recvd)
            recvd = socket.recv(buffer_size)
        file.write(recvd)
    data = pickle.load(open(filename, 'rb'))
    return data

def unpickle_data(filename):
    data = pickle.load(open(f'{filename}.pkl', 'rb'))
    return data

class ClientModel(nn.Module):
    def __init__(self):
        super(ClientModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        #x = x.view(-1, 16 * 4 * 4)
        return x
    
class ServerModel(nn.Module):
    def __init__(self):
        super(ServerModel, self).__init__()
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class ClientClassifier(nn.Module):
    def __init__(self):
        super(ClientClassifier, self).__init__()
        self.fc1 = nn.Linear(16 * 4 * 4, 10)

    def forward(self, x):
        x = x.view(-1, 16 * 4 * 4)
        x = self.fc1(x)
        return x
    

def train_one_epoch(epoch_index, training_loader, optimizer, loss_fn, servermodel, clientsocket, client_batches, client_address):
    running_loss = 0.
    last_loss = 0.
    print('one epoch mpika')
    print('Client batches: ', client_batches)
    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i in range(client_batches):
        # Every data instance is an input + label pair
        #inputs, labels = data

        # Zero your gradients for every batch!
        optimizer.zero_grad()
        #if i == 2:
            #print('End of training')
            #break
        # Make predictions for this batch
        #outputs1 = model1(inputs)
        event.wait()
        #with file_condition:
        #file_condition.wait()
        with file_lock:
            client_sent = unpickle_data(f'{client_address}')
        event.clear()
        #print('Weights Recieved')
        #client_sent = pickle.load(open('client_outputs.pkl', 'rb'))
        outputs2 = servermodel(client_sent[0])
        # Compute the loss and its gradients send client backprop
        loss = loss_fn(outputs2, client_sent[1])
        global_loss = (0.5 * client_sent[2]) + (0.5 * loss)
        #pickle.dump(global_loss, open('global_loss_sent.pkl', 'wb'))
        send_data('global_loss_sent.pkl', clientsocket, global_loss)
        global_loss.backward()

        # Adjust learning weights
        optimizer.step()
        
        # Gather data and report
        running_loss += loss.item()
        if i % 1000 == 999:
            last_loss = running_loss / 1000 # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            running_loss = 0.
        
    return last_loss

def train(epochs, server_model, train_dataloader, valid_dataloader, optimizer, loss_fn, clientsocket, client_address):

    best_vloss = 1_000_000.

    for epoch in range(epochs):
        print(f'Epoch {epoch + 1} :')
        event.wait()
        #with file_condition:
        #file_condition.wait()
        with file_lock:
            client_batches = unpickle_data(f'{client_address}')
        event.clear()
        #client_batches = pickle.load(open('client_batches.pkl', 'rb'))
        print('Client_batches :', client_batches)
        # Make sure gradient tracking is on, and do a pass over the data
        server_model.train()
        avg_loss = train_one_epoch(epoch, train_dataloader, optimizer, loss_fn, server_model, clientsocket, client_batches, client_address)

        # We don't need gradients on to do reporting
        server_model.eval()
        #client_batches = receive_data('client_batches.pkl', clientsocket)
        event.wait()
        #with file_condition:
            #file_condition.wait()
        with file_lock:
            client_batches = unpickle_data(f'{client_address}')
        event.clear()
        #print('Val Batches: ', client_batches)
        running_vloss = 0.0
        for batch in range(client_batches):
            #print('mpika')
            #client_sent = receive_data('client_valid_recvd.pkl', clientsocket)
            event.wait()
            #with file_condition:
                #file_condition.wait()
            with file_lock:
                client_sent = unpickle_data(f'{client_address}')
            event.clear()
            print(len(client_sent))
            voutputs1 = server_model(client_sent[0])
            vloss = loss_fn(voutputs1, client_sent[1])
            running_vloss += vloss

        avg_vloss = running_vloss / (batch + 1)
        print(f'Average Training Loss: {avg_loss: .3f}')
        print(f'Average Validation Loss: {avg_vloss: .3f}')


        # Track best performance, and save the model's state
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            #model_path = 'model_{}_{}'.format(timestamp, epoch_number)
            #torch.save(model.state_dict(), model_path)

def receive_data1(data, client_address):
    with open(f'{client_address}_data.pkl', 'wb') as file:
        file.write(data)
    return


def listen_for_data(client_socket, client_address):
    print(f'[+] Communication thread for {client_address} created.')
    data_queue = queue.Queue()
    while True:
        data = client_socket.recv(4096)
        data_queue.put(data)
        #receive_data1(data, client_address)
        #print(data)
        if str(data).__contains__('Done'):
            data_queue.put('SEPERATOR')
            storing_thread = threading.Thread(target=store_data, args=(client_address, data_queue, client_socket))
            storing_thread.start()
        if not data:
            break

def store_data(client_address, data_queue, client_socket):
    #with file_condition:
    with file_lock:
        with open(f'{client_address}.pkl', 'wb') as file:
            data = data_queue.get()
            while data != 'SEPERATOR':
                file.write(data)
                data = data_queue.get()
    client_socket.send(b'Done')
    event.set()
        #file_condition.notify_all()
    return


def listen_for_connections(socket):
    while True:
        client_socket, client_address = socket.accept()
        print(f'[+] Connection with {client_address} established.')
        communication_thread = threading.Thread(target=listen_for_data, args=(client_socket, client_address))
        communication_thread.start()
        client_thread = threading.Thread(target=client_handler, args=(client_socket, client_address))
        client_thread.start()

def client_handler(client_socket, client_address):
    client_model = ClientModel()
    server_model = ServerModel()
    send_data(f'{client_address}_data.pkl', client_socket, client_model.state_dict())
    print('Weights sent')
    device = get_default_device()
    device

    dataset, test_dataset = get_dataset()
    train_dl, valid_dl, test_dl = create_dataloader(dataset, test_dataset)

    loss_fn = torch.nn.CrossEntropyLoss()
    # Optimizers specified in the torch.optim package
    optimizer = torch.optim.SGD(server_model.parameters(), lr=0.001, momentum=0.9)
    EPOCHS = 2
    
    train(EPOCHS, server_model, train_dl, valid_dl, optimizer, loss_fn, client_socket, client_address)
#training klp edw


def main():

    # Socket Creation and Connection
    server = create_socket_and_listen(serverip, serverport)
    
    
#------------------AUTA PANW MENOUN EDW------------------
"""
    print('Weights: ', client_model.state_dict())

    # Sending the initial weights to the client
    #pickle.dump(client_model.state_dict(), open('server_weights.pkl', 'wb'))
    send_data('server_weights.pkl', clientsocket, client_model.state_dict())
    print('Weights sent')


    # Device agnostic code
    device = get_default_device()
    device

    dataset, test_dataset = get_dataset()
    train_dl, valid_dl, test_dl = create_dataloader(dataset, test_dataset)

    loss_fn = torch.nn.CrossEntropyLoss()
    # Optimizers specified in the torch.optim package
    optimizer = torch.optim.SGD(server_model.parameters(), lr=0.001, momentum=0.9)
    EPOCHS = 2
    
    train(EPOCHS, server_model, train_dl, valid_dl, optimizer, loss_fn, clientsocket)
"""
    
if __name__ == '__main__':
    main()