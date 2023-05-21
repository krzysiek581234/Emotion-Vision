import torch.nn as nn
class NeuralNet(nn.Module):
    # Relu6 test acc = 47.1161
    def __init__(self, input_size, num_classes):

        hidden_size1 = 1024
        hidden_size2 = 512
        hidden_size3 = 256
        hidden_size4 = 128
        super(NeuralNet, self).__init__()
        self.input_size = input_size
        self.dropout = nn.Dropout(p=0.1)
        self.relu = nn.LeakyReLU()
        #magic
        self.l1 = nn.Linear(input_size, hidden_size1)
        nn.init.xavier_uniform_(self.l1.weight)
        self.b1 = nn.BatchNorm1d(hidden_size1)

        self.l2 = nn.Linear(hidden_size1, hidden_size2)
        nn.init.xavier_uniform_(self.l2.weight)
        self.b2 = nn.BatchNorm1d(hidden_size2)

        self.l3 = nn.Linear(hidden_size2, hidden_size3)
        nn.init.xavier_uniform_(self.l3.weight)
        self.b3 = nn.BatchNorm1d(hidden_size3)

        self.l4 = nn.Linear(hidden_size3, hidden_size4)
        nn.init.xavier_uniform_(self.l4.weight)
        self.b4 = nn.BatchNorm1d(hidden_size4)

        self.l5 = nn.Linear(hidden_size4, num_classes)
        nn.init.xavier_uniform_(self.l5.weight)

    def forward(self, x):
        x = self.l1(x)

        x = self.b1(x)
        x = self.relu(x)

        x = self.dropout(x)
        x = self.l2(x)
        x = self.b2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.l3(x)
        x = self.b3(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.l4(x)
        x = self.b4(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.l5(x)
        return x