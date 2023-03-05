import pyodbc
import socket
import pickle
import threading
import torch
from torch import nn as nn
import struct
import torch.optim as optim
import torch.nn.functional as F
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
training_lock = threading.Lock()
cursor_lock = threading.Lock()
select_event = threading.Event()

class Server:
    def __init__(self):
        ##super().__init__()
        self.ip = 'localhost'
        self.port = 9999
        self.fetching_connection = pyodbc.connect(
            'Driver={ODBC Driver 17 for SQL Server};'
            'Server=LAPTOP-LGNU4S88;'
            'Database=Thesis;'
            'Trusted_Connection=yes;'
        )
        self.storing_connection = pyodbc.connect(
            'Driver={ODBC Driver 17 for SQL Server};'
            'Server=LAPTOP-LGNU4S88;'
            'Database=Thesis;'
            'Trusted_Connection=yes;'
        )
        self.storing_cursor = self.storing_connection.cursor()
        self.fetching_cursor = self.fetching_connection.cursor()
        self.client_counter = 0
        threading.Thread(target=self.create_socket, args=(())).start()

    def create_socket(self):
        try:
            server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            print(f'[+] Server socket created successfully at {self.ip, self.port}')
            server.bind((self.ip, self.port))
            server.listen()
            listen_thread = threading.Thread(target=self.listen_for_connections, args=(server,))
            listen_thread.start()
        except socket.error as error:
            print(f'Socket creation failed with error: {error}')
            server.close()

    def send_data(self, socket, data):
        serialized_tensor = pickle.dumps(data)
        #print(len(serialized_tensor))
        socket.sendall(serialized_tensor)
        socket.sendall(b'<Done>')
        return
    
    def listen_for_connections(self, server):
        while True:
            client_socket, client_address = server.accept()
            print(f'[+] Connection with {client_address} established.')
            with cursor_lock:
                self.storing_cursor.execute('INSERT INTO clients(address, port) VALUES (?, ?)', (client_address[0], client_address[1]))
                self.storing_connection.commit()
                self.storing_cursor.execute('SELECT id FROM clients WHERE address = ? AND port = ?', client_address[0], client_address[1])
                client_id = self.storing_cursor.fetchone()[0]
            self.client_counter += 1
            #print('Added to table')
            communication_thread = threading.Thread(target=self.listen_for_data, args=(client_socket, client_address, client_id))
            communication_thread.start()
            client_thread = threading.Thread(target=self.client_handler, args=(client_socket, client_address, client_id))
            client_thread.start()
        
    def listen_for_data(self, client_socket, client_address, client_id):
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
                    set = 'training' if header.__contains___('train') else 'validation'
                    storing_thread = threading.Thread(target=self.store_data, args=(client_address, data, header, True, set, client_id))
                    storing_thread.start()
                else:
                    storing_thread = threading.Thread(target=self.store_data, args=(client_address, data, header, False, set, client_id))
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
                    set = 'training' if header.__contains__('train') else 'validation'
                    storing_thread = threading.Thread(target=self.store_data, args=(client_address, data, header, True, set, client_id))
                    storing_thread.start()
                else:
                    storing_thread = threading.Thread(target=self.store_data, args=(client_address, data, header, False, set, client_id))
                    storing_thread.start()
                client_socket.send(bytes('<Recvd>', 'utf-8'))
                data = b''
            else:
                #print('Bainw akyra')
                data += data_chunk

            if not data_chunk:
                break

    def store_data(self, client_address, data, header, int_flag, set, client_id):
        query = f"""
                IF EXISTS (SELECT * FROM {set} WHERE client_id = ?)
                BEGIN
                    UPDATE {set} SET {header} = ? WHERE client_id = ?
                END
                ELSE
                BEGIN
                    INSERT INTO {set} (client_id, {header}) VALUES (?, ?)
                END"""
        if int_flag:
            with cursor_lock:
                self.storing_cursor.execute(query, (client_id, int(struct.unpack('!i', data)[0]), client_id, client_id, int(struct.unpack('!i', data)[0])))
        else:
            with cursor_lock:
                self.storing_cursor.execute(query, (client_id, data, client_id, client_id, data))
        
        self.storing_connection.commit()

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

    def client_handler(self, client_socket, client_address, client_id):
        # Model Initialization
        client_model = ClientModel()
        server_model = ServerModel()
        self.send_data(client_socket, client_model.state_dict())
        #state_dict_bytes = pickle.dumps(client_model.state_dict())
        #print(len(state_dict_bytes))
        #client_socket.sendall(state_dict_bytes)
        #client_socket.send(b'Done')
        print('Weights sent')
        # Device agnostic code
        device = self.get_default_device()
        device
        # Define loss function and optimizer
        loss_fn = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(server_model.parameters(), lr=0.001, momentum=0.9)
        # Start Training
        EPOCHS = 20
        self.train(EPOCHS, server_model, optimizer, loss_fn, client_socket, client_address, client_id)


    def get_default_device(self, ):
        if torch.cuda.is_available():
            return torch.device('cuda')
        else:
            return torch.device('cpu')

    def to_device(self, data, device):
        if isinstance(data, (list, tuple)):
            return [self.to_device(x, device) for x in data]
        return data.to(device, non_blocking=True)
    #train_one_epoch(epoch, optimizer, loss_fn, server_model, clientsocket, client_batches, client_address)
    def train_one_epoch(self, optimizer, loss_fn, servermodel, clientsocket, client_batches, client_address, client_id):
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
                self.fetching_cursor.execute('SELECT training_outputs FROM training WHERE client_id = ?', client_id)
                select_event.set()
                select_event.wait()
                select_event.clear()
                client_outputs = pickle.loads(self.fetching_cursor.fetchone()[0])
                self.fetching_connection.commit()
                self.fetching_cursor.close()
                self.fetching_cursor = self.fetching_connection.cursor()
            training_outputs_event.clear()
            #print('Elava client outputs')
            outputs2 = servermodel(client_outputs)
            #print('Ypologisa ta dika mou outputs')
            # Compute the loss and its gradients send client backprop
            training_labels_event.wait()
            with cursor_lock:
                self.fetching_cursor.execute('SELECT training_labels FROM training WHERE client_id = ?', client_id)
                select_event.set()
                select_event.wait()
                select_event.clear()
                client_labels = pickle.loads(self.fetching_cursor.fetchone()[0])
                self.fetching_connection.commit()
                self.fetching_cursor.close()
                self.fetching_cursor = self.fetching_connection.cursor()
            training_labels_event.clear()
            loss = loss_fn(outputs2, client_labels)
            training_loss_event.wait()
            #print('Ypologisa loss mou')
            with cursor_lock:
                self.fetching_cursor.execute('SELECT training_loss FROM training WHERE client_id = ?', client_id)
                select_event.set()
                select_event.wait()
                select_event.clear()
                client_loss = pickle.loads(self.fetching_cursor.fetchone()[0])
                self.fetching_connection.commit()
                self.fetching_cursor.close()
                self.fetching_cursor = self.fetching_connection.cursor()
            training_loss_event.clear()
            global_loss = (0.5 * client_loss) + (0.5 * loss)
            self.send_data(clientsocket, global_loss)
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
        return last_loss

    def train(self, epochs, server_model, optimizer, loss_fn, clientsocket, client_address, client_id):

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
                self.fetching_cursor.execute(f'SELECT training_batches FROM training WHERE client_id = ?', client_id)
                select_event.set()
                select_event.wait()
                select_event.clear()
                client_batches = self.fetching_cursor.fetchone()[0]
                self.fetching_connection.commit()
                self.fetching_cursor.close()
                self.fetching_cursor = self.fetching_connection.cursor()
            training_batches_event.clear()
            print('Client batches :', client_batches)
            server_model.train()
            avg_loss = self.train_one_epoch(optimizer, loss_fn, server_model, clientsocket, client_batches, client_address, client_id)
            
            # We don't need gradients on to do reporting
            server_model.eval()
            validation_batches_event.wait()
            validation_batches_event.clear()
            with cursor_lock:
                self.fetching_cursor.execute(f'SELECT validation_batches FROM validation WHERE client_id = ?', client_id)
                select_event.set()
                select_event.wait()
                select_event.clear()
                client_batches = self.fetching_cursor.fetchone()[0]
                self.fetching_connection.commit()
                self.fetching_cursor.close()
                self.fetching_cursor = self.fetching_connection.cursor()
            running_vloss = 0.0
            print('Validation Batches: ', client_batches)
            for batch in range(client_batches):
                validation_outputs_event.wait()
                validation_outputs_event.clear()
                with cursor_lock:
                    self.fetching_cursor.execute(f'SELECT validation_outputs FROM validation WHERE client_id = ?', client_id)
                    select_event.set()
                    select_event.wait()
                    select_event.clear()
                    client_outputs = pickle.loads(self.fetching_cursor.fetchone()[0])
                    self.fetching_connection.commit()
                self.fetching_cursor.close()
                self.fetching_cursor = self.fetching_connection.cursor()
                validation_labels_event.wait()
                validation_labels_event.clear()
                with cursor_lock:
                    self.fetching_cursor.execute(f'SELECT validation_labels FROM validation WHERE client_id = ?', client_id)
                    select_event.set()
                    select_event.wait()
                    select_event.clear()
                    validation_labels = pickle.loads(self.fetching_cursor.fetchone()[0])
                    self.fetching_connection.commit()
                    self.fetching_cursor.close()
                    self.fetching_cursor = self.fetching_connection.cursor()
                voutputs1 = server_model(client_outputs)
                vloss = loss_fn(voutputs1, validation_labels)
                clientsocket.send(b'<OK>')
                running_vloss += vloss


            
            avg_vloss = running_vloss / (batch + 1)
            print(f'Average Training Loss: {avg_loss: .3f}')
            print(f'Average Validation Loss: {avg_vloss: .3f}')


            # Track best performance, and save the model's state
            if avg_vloss < best_vloss:
                best_vloss = avg_vloss
                #model_path = 'model_{}_{}'.format(timestamp, epoch_number)
                #torch.save(model.state_dict(), model_path)
        #training_event.set()


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
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    

if __name__ == '__main__':
    Server()