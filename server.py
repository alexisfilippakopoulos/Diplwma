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
import struct
import pyodbc
import numpy as np
from tqdm.auto import tqdm


training_batches_event = threading.Event()
training_outputs_event = threading.Event()
training_labels_event = threading.Event()
training_loss_event = threading.Event()
validation_batches_event = threading.Event()
validation_outputs_event = threading.Event()
validation_labels_event = threading.Event()

event = threading.Event()
training_event = threading.Event()
training_event.set()
file_lock = threading.Lock()
training_lock = threading.Lock()
cursor_lock = threading.Lock()
select_event = threading.Event()
file_condition = threading.Condition()
serverip = 'localhost'
serverport = 9999
buffer_size = 4096
connection = pyodbc.connect(
    'Driver={ODBC Driver 17 for SQL Server};'
    'Server=LAPTOP-LGNU4S88;'
    'Database=SplitGP;'
    'Trusted_Connection=yes;'

)
storing_connection = pyodbc.connect(
    'Driver={ODBC Driver 17 for SQL Server};'
    'Server=LAPTOP-LGNU4S88;'
    'Database=SplitGP;'
    'Trusted_Connection=yes;'

)

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
        storing_cursor = storing_connection.cursor()
        fetching_cursor = connection.cursor()
        listen_thread = threading.Thread(target=listen_for_connections, args=(server, storing_cursor, fetching_cursor))
        listen_thread.start()
        return server
    except socket.error as err:
        print(f"Socket creation failed with error {err}")

def send_data(socket, data):
    serialized_tensor = pickle.dumps(data)
    #print(len(serialized_tensor))
    socket.sendall(serialized_tensor)
    socket.sendall(b'<Done>')
    return
    """
    pickle.dump(data, open(filename, 'wb'))
    buffer = 4096
    with open(filename, 'rb') as file:
        for buffer in file:
            socket.send(buffer)
    socket.send(b"Done")
    return
    """

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
        return x
    
class ServerModel(nn.Module):
    def __init__(self):
        super(ServerModel, self).__init__()
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        if isinstance(x, bytes):
            print('EINAI BYTE TOMPE')
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
    

def train_one_epoch(epoch_index, training_loader, optimizer, loss_fn, servermodel, clientsocket, client_batches, client_address, fetching_cursor):
    running_loss = 0.
    last_loss = 0.
    #print('Client training batches: ', client_batches)
    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i in range(client_batches):
        #print('Mpika sto per epoch')
        # Zero your gradients for every batch!
        optimizer.zero_grad()
        #if i == 2:
            #print('End of training')
            #break
        # Make predictions for this batch
        training_outputs_event.wait()
        with cursor_lock:
            fetching_cursor.execute('SELECT training_outputs FROM clients WHERE socket = ?', (client_address[1],))
            select_event.set()
            select_event.wait()
            select_event.clear()
            client_outputs = pickle.loads(fetching_cursor.fetchone()[0])
            connection.commit()
            fetching_cursor.close()
            fetching_cursor = connection.cursor()
        training_outputs_event.clear()
        #print('Elava client outputs')
        outputs2 = servermodel(client_outputs)
        #print('Ypologisa ta dika mou outputs')
        # Compute the loss and its gradients send client backprop
        training_labels_event.wait()
        with cursor_lock:
            fetching_cursor.execute('SELECT training_labels FROM clients WHERE socket = ?', (client_address[1],))
            select_event.set()
            select_event.wait()
            select_event.clear()
            client_labels = pickle.loads(fetching_cursor.fetchone()[0])
            connection.commit()
            fetching_cursor.close()
            fetching_cursor = connection.cursor()
        training_labels_event.clear()
        loss = loss_fn(outputs2, client_labels)
        training_loss_event.wait()
        #print('Ypologisa loss mou')
        with cursor_lock:
            fetching_cursor.execute('SELECT training_loss FROM clients WHERE socket = ?', (client_address[1],))
            select_event.set()
            select_event.wait()
            select_event.clear()
            client_loss = pickle.loads(fetching_cursor.fetchone()[0])
            connection.commit()
            fetching_cursor.close()
            fetching_cursor = connection.cursor()
        training_loss_event.clear()
        global_loss = (0.5 * client_loss) + (0.5 * loss)
        send_data(clientsocket, global_loss)
        #print('global loss: ', global_loss)
        global_loss.backward()

        # Adjust learning weights
        optimizer.step()
        
        # Gather data and report
        running_loss += loss.item()
        if i % 1000 == 999:
            last_loss = running_loss / 1000 # loss per batch
            print(f'Batch: {i + 1} ')
            print(f'    Training Loss: {last_loss}')
            print(f'    Global Loss: {global_loss}')
            running_loss = 0.
    #print('eftasa sto na fyge apo per epoch')
    return last_loss, fetching_cursor

def train(epochs, server_model, train_dataloader, valid_dataloader, optimizer, loss_fn, clientsocket, client_address, fetching_cursor):

    best_vloss = 1_000_000.
    training_event.wait()
    clientsocket.send(b'<train>')
    print(f'Training with {client_address} intiated')
    training_event.clear()
    for epoch in tqdm(range(epochs), desc="Epoch"):
        #print(f'Epoch {epoch + 1} :')
        training_batches_event.wait()
        #with file_lock:
            #client_batches = unpickle_data(f'{client_address}')
        with cursor_lock:
            fetching_cursor.execute(f'SELECT training_batches FROM clients WHERE socket = {client_address[1]}')
            select_event.set()
            select_event.wait()
            select_event.clear()
            client_batches = fetching_cursor.fetchone()[0]
            connection.commit()
            fetching_cursor.close()
            fetching_cursor = connection.cursor()
        training_batches_event.clear()
        print('Client batches :', client_batches)
        server_model.train()
        avg_loss, fetching_cursor = train_one_epoch(epoch, train_dataloader, optimizer, loss_fn, server_model, clientsocket, client_batches, client_address, fetching_cursor)
        
        # We don't need gradients on to do reporting
        server_model.eval()
        validation_batches_event.wait()
        validation_batches_event.clear()
        with cursor_lock:
            fetching_cursor.execute(f'SELECT validation_batches FROM clients WHERE socket = {client_address[1]}')
            select_event.set()
            select_event.wait()
            select_event.clear()
            client_batches = fetching_cursor.fetchone()[0]
            connection.commit()
            fetching_cursor.close()
            fetching_cursor = connection.cursor()
        running_vloss = 0.0
        print('Validation Batches: ', client_batches)
        for batch in range(client_batches):
            validation_outputs_event.wait()
            validation_outputs_event.clear()
            with cursor_lock:
                fetching_cursor.execute(f'SELECT validation_outputs FROM clients WHERE socket = {client_address[1]}')
                select_event.set()
                select_event.wait()
                select_event.clear()
                client_outputs = pickle.loads(fetching_cursor.fetchone()[0])
                connection.commit()
            fetching_cursor.close()
            fetching_cursor = connection.cursor()
            #print(client_outputs.shape)
            validation_labels_event.wait()
            validation_labels_event.clear()
            with cursor_lock:
                fetching_cursor.execute(f'SELECT validation_labels FROM clients WHERE socket = {client_address[1]}')
                select_event.set()
                select_event.wait()
                select_event.clear()
                validation_labels = pickle.loads(fetching_cursor.fetchone()[0])
                connection.commit()
                fetching_cursor.close()
                fetching_cursor = connection.cursor()
            voutputs1 = server_model(client_outputs)
            vloss = loss_fn(voutputs1, validation_labels)
            clientsocket.send(b'<OK>')
            #print('Esteiila OK')
            running_vloss += vloss
            #eaprint(batch)

        
        avg_vloss = running_vloss / (batch + 1)
        print(f'Average Training Loss: {avg_loss: .3f}')
        print(f'Average Validation Loss: {avg_vloss: .3f}')


        # Track best performance, and save the model's state
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            #model_path = 'model_{}_{}'.format(timestamp, epoch_number)
            #torch.save(model.state_dict(), model_path)
    #training_event.set()
def receive_data1(data, client_address):
    with open(f'{client_address}_data.pkl', 'wb') as file:
        file.write(data)
    return


def listen_for_data(client_socket, client_address, storing_cursor):
    print(f'[+] Communication thread for {client_address} created.')
    data = b''
    while True:
        data_chunk = client_socket.recv(4096)
        #print('Arxiko chunk: ', data_chunk)
        if str(data_chunk).__contains__('<SEPERATOR>') and str(data_chunk).__contains__('Done'):
            #print('Mpika stin kaki')
            payload = data_chunk.split(b'<SEPERATOR>')[1].split(b'Done')[0]
            payload = bytes(payload)
            data += payload
            if header.__contains__('batch'):
                #print('stlenw thread gia int')
                storing_thread = threading.Thread(target=store_data, args=(client_address, data, header, storing_cursor, True))
                storing_thread.start()
            else:
                storing_thread = threading.Thread(target=store_data, args=(client_address, data, header, storing_cursor, False))
                storing_thread.start()
            
            #print(f'Dexthika {header}')
            client_socket.send(bytes('<Recvd>', 'utf-8'))
            data = b''
        elif str(data_chunk).__contains__('<SEPERATOR>'):
            #print('Chunk sto sep: ', data_chunk)
            header, payload = data_chunk.split(b'<SEPERATOR>', 1)
            header = header.decode()
            #print('Header sto sep: ', header)
            #print('Payload sto sep: ', payload)
            if payload != b'':
                #print('mpika')
                payload = bytes(payload)
                data += payload
            #pass
            #print('Data sto sep: ', data)
        elif str(data_chunk).__contains__('<Done>'):
            payload, _ = data_chunk.split(b'<Done>', 1)
            #print('Payload sto done: ', payload)
            payload = bytes(payload)
            #print('Payload sto done: ', payload)
            data += payload
            #print('Data to send to storage: ', len(data))
            if header.__contains__('batch'):
                #print('stlenw thread gia int')
                storing_thread = threading.Thread(target=store_data, args=(client_address, data, header, storing_cursor, True))
                storing_thread.start()
            else:
                storing_thread = threading.Thread(target=store_data, args=(client_address, data, header, storing_cursor, False))
                storing_thread.start()
            client_socket.send(bytes('<Recvd>', 'utf-8'))
            data = b''
        else:
            #print('Bainw akyra')
            data += data_chunk

        if not data_chunk:
            break


def store_data(client_address, data, header, storing_cursor, int_flag):
    if int_flag:
        with cursor_lock:
            storing_cursor.execute(f"UPDATE clients SET {header} = ? WHERE socket = ?", (struct.unpack('!i', data)[0], int(client_address[1])))
    else:
        with cursor_lock:
            storing_cursor.execute(f"UPDATE clients SET {header} = ? WHERE socket = ?", (data, int(client_address[1])))
    
    storing_connection.commit()
    if header == 'training_batches': 
        training_batches_event.set()
    if header == 'training_outputs': 
        training_outputs_event.set()
    if header == 'training_labels': 
        training_labels_event.set()
    if header == 'training_loss': 
        training_loss_event.set()
    if header == 'validation_batches': 
        validation_batches_event.set()
    if header == 'validation_outputs': 
        validation_outputs_event.set()
    if header == 'validation_labels': 
        validation_labels_event.set()
    
    
    #event.set()
    return


def listen_for_connections(socket, storing_cursor, fetching_cursor):
    while True:
        client_socket, client_address = socket.accept()
        print(f'[+] Connection with {client_address} established.')
        with cursor_lock:
            storing_cursor.execute('INSERT INTO clients(socket) VALUES (?)', (client_address[1],))
        connection.commit()
        #print('Added to table')
        communication_thread = threading.Thread(target=listen_for_data, args=(client_socket, client_address, storing_cursor))
        communication_thread.start()
        client_thread = threading.Thread(target=client_handler, args=(client_socket, client_address, fetching_cursor))
        client_thread.start()

def client_handler(client_socket, client_address, fetching_cursor):
    # Model Initialization
    client_model = ClientModel()
    server_model = ServerModel()
    send_data(client_socket, client_model.state_dict())
    #state_dict_bytes = pickle.dumps(client_model.state_dict())
    #print(len(state_dict_bytes))
    #client_socket.sendall(state_dict_bytes)
    #client_socket.send(b'Done')
    print('Weights sent')
    # Device agnostic code
    device = get_default_device()
    device
    # Get dataset and dataloaders
    dataset, test_dataset = get_dataset()
    train_dl, valid_dl, test_dl = create_dataloader(dataset, test_dataset)
    # Define loss function and optimizer
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(server_model.parameters(), lr=0.001, momentum=0.9)
    # Start Training
    EPOCHS = 20
    train(EPOCHS, server_model, train_dl, valid_dl, optimizer, loss_fn, client_socket, client_address, fetching_cursor)



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