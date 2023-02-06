import socket
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision
from torch.utils.data import DataLoader, random_split
import pickle
import struct
import threading

host = '127.0.0.1'
port = 9999
buffer_size = 4096
recvd_event = threading.Event()
train_event = threading.Event()
server_completion = threading.Event()
received_event = threading.Event()

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

class ClientClassifier(nn.Module):
    def __init__(self):
        super(ClientClassifier, self).__init__()
        self.fc1 = nn.Linear(16 * 4 * 4, 10)

    def forward(self, x):
        x = x.view(-1, 16 * 4 * 4)
        x = self.fc1(x)
        return x

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
    batch_size = 32
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

def train_one_epoch(epoch_index, training_loader, optimizer, loss_fn, model1, model2, server):
    running_loss = 0.
    last_loss = 0.
    #print('one epoch mpika')
    print(len(training_loader))
    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, data in enumerate(training_loader):
        # 15000 fores ginete auto to loop per epoch
        # Every data instance is an input + label pair
        #if i ==2:
            #print('End of training')
            #break
        inputs, labels = data
        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs1 = model1(inputs)
        #print(outputs1.shape)
        
        outputs2 = model2(outputs1)

        # Compute the loss and its gradients
        loss = loss_fn(outputs2, labels)
        #pickle.dump([outputs1, labels, loss], open('epoch_data.pkl', 'wb'))
        #send_data('epoch_data', server, [outputs1, labels, loss])
        send_data('training_outputs', server, outputs1)
        received_event.wait()
        received_event.clear()
        send_data('training_labels', server, labels)
        received_event.wait()
        received_event.clear()
        send_data('training_loss', server, loss)
        received_event.wait()
        received_event.clear()
        recvd_event.wait()
        global_loss = pickle.load(open('server_sent.pkl', 'rb'))
        recvd_event.clear()
        #print('Global_loss', global_loss)
        #global_loss = pickle.load(open('global_loss_recvd.pkl', 'rb'))
        # recv tensor compute new loss backprop
        global_loss.backward()

        # Adjust learning weights
        optimizer.step()
        
        # Gather data and report
        running_loss += loss.item()
        if i % 1000 == 999:
            last_loss = running_loss / 1000 # loss per batch
            print(f'Batch: {i + 1} ')
            print(f'    Training Loss: {last_loss}')
            running_loss = 0.
        
    return last_loss

def train(epochs, model1, model2, train_dataloader, valid_dataloader, optimizer, loss_fn, server):


    #train_event.wait()
    #train_event.clear()
    best_vloss = 1_000_000.
    for epoch in range(epochs):
        print(f'Epoch {epoch + 1} :')
            #print(len(train_dataloader))
            #pickle.dump(len(train_dataloader), open('batches_num.pkl', 'wb'))
        send_data('training_batches', server, len(train_dataloader))
        received_event.wait()
        received_event.clear()
        print('ayo')
                # Make sure gradient tracking is on, and do a pass over the data
        model1.train()
        model2.train()
        avg_loss = train_one_epoch(epoch, train_dataloader, optimizer, loss_fn, model1, model2, server)

                # We don't need gradients on to do reporting
        model1.eval()
        model2.eval()
        
        running_vloss = 0.0

        send_data('validation_batches', server, len(valid_dataloader))
        received_event.wait()
        received_event.clear()
        for i, vdata in enumerate(valid_dataloader):
            server_completion.set() if i == 0 else None
                    #print('mpika')
            vinputs, vlabels = vdata
            voutputs1 = model1(vinputs)
            server_completion.wait()
            server_completion.clear()
            send_data('validation_outputs', server, voutputs1)
            #print('val shape', voutputs1.shape)
            received_event.wait()
            received_event.clear()
            send_data('validation_labels', server, vlabels)
            received_event.wait()
            received_event.clear()
            voutputs2 = model2(voutputs1)
                    #print('val data sent')
                    #print(i)
            vloss = loss_fn(voutputs2, vlabels)
            running_vloss += vloss

        avg_vloss = running_vloss / (i + 1)
        print(f'Average Training Loss: {avg_loss: .3f}')
        print(f'Average Validation Loss: {avg_vloss: .3f}')


                # Track best performance, and save the model's state
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
                    #model_path = 'model_{}_{}'.format(timestamp, epoch_number)
                    #torch.save(model.state_dict(), model_path)

def send_data(message, socket, data):
    message += '<SEPERATOR>'
    #print('Message to send: ', message)
    socket.send(bytes(message, 'utf-8'))
    if isinstance(data, int):
        data = struct.pack('!i', data)
        socket.send(data)
        socket.send(b'Done')
    else:
        serialized_tensor = pickle.dumps(data)
        socket.sendall(serialized_tensor)
        socket.sendall(b'Done')
    return
    """
    pickle.dump(data, open(filename, 'wb'))
    buffer = 4096
    with open(filename, 'rb') as file:
        for buffer in file:
        #print(buffer)
            socket.send(buffer)
    socket.send(b"Done")
    data = socket.recv(1024)
    if str(data).__contains__('Done'):
        return
    else:
        print('ZAAAAMN')
    """

def receive_data(data):
    with open('server_sent.pkl', 'wb') as file:
        pickle.dump(data, file)
    recvd_event.set()
    return

def create_socket_and_connect(host, port):
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        print(f"[+] Connecting to {host}:{port}")
        s.connect((host, port))
        print("[+] Connected.")
        listen_thread = threading.Thread(target=listen_for_data, args=(s, ))
        listen_thread.start()
        return s
    except socket.error as err:
        print(f"Socket creation failed with error {err}")

def listen_for_data(server):
    print(f'[+] Communication thread for {server} created.')
    data = b''
    while True:
        data_chunk = server.recv(4096)
        #print(data_chunk)
        if str(data_chunk).__contains__('Done'):
            payload, _ = data_chunk.split(b'Done', 1)
            payload = bytes(payload)
            data += payload
            #print(len(data))
            #print(str(data).__contains__('Done'))
            #print(str(data).__contains__('train'))
            #print(str(data).__contains__('Recvd'))
            #store_thread = threading.Thread(target=receive_data, args=(data, ))
            #store_thread.start()
            with open('server_sent.pkl', 'wb') as file:
                file.write(data)
            recvd_event.set()
            data = b''
        elif data_chunk == b'train':
            train_event.set()
        elif data_chunk == b'Recvd':
            received_event.set()
        elif data_chunk == b'OK':
            server_completion.set()
        else:
            data += data_chunk


def main():

    client_model = ClientModel()
    client_classifier = ClientClassifier()

    server = create_socket_and_connect(host, port)
    recvd_event.wait()
    server_weights = pickle.load(open('server_sent.pkl', 'rb'))
    recvd_event.clear()
    print('Recieved Weights')
    #server_weights = pickle.load(open('starting_weights.pkl', 'rb'))
    #print(f'Server Weights: {server_weights}')
    client_model.load_state_dict(server_weights)
    print(client_model.state_dict())

    # Device agnostic code
    device = get_default_device()
    device

    dataset, test_dataset = get_dataset()
    train_dl, valid_dl, test_dl = create_dataloader(dataset, test_dataset)

    loss_fn = torch.nn.CrossEntropyLoss()
    # Optimizers specified in the torch.optim package
    optimizer = torch.optim.SGD(client_model.parameters(), lr=0.001, momentum=0.9)


    EPOCHS = 20

    train(EPOCHS, client_model, client_classifier, train_dl, valid_dl, optimizer, loss_fn, server)


if __name__ == '__main__':
    main()