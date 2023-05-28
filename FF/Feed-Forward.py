import os

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from NeuralNet import NeuralNet
from readDataFCNN import readDataFCNN
from tensorboardX import SummaryWriter
#device name
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
log_dir = 'D:/Studia_Krzysiek/Semestr4/Projekty/Projekt_Sztuczna/Emotion-Vision/logs/FF'
# tensorboard --logdir=D:/Studia_Krzysiek/Semestr4/Projekty/Projekt_Sztuczna/Emotion-Vision/logs/FF
#learning_rateArray = [0.001]
learning_rateArray = [0.0005,0.001,0.0015, 0.002,0.004]
batchSizeArray = [16,32,64,128,256]

Leaning_csv = '../dataset/LearningTest.csv'
test_csv_file = '../dataset/PublicTest.csv'

result = [[0.0] * 5 for _ in range(5)]
#learing rate Osi X
#batch size rate Osi Y

for x, batch_size in enumerate(batchSizeArray):
    for y, learning_rate in enumerate(learning_rateArray):
        for i in range(len(result)):
            for j in range(len(result[i])):
                print(result[i][j], end=" ")
            print()
        # Create a subdirectory for each run based on j and i
        run_dir = f'run_batch{batch_size}LR_{learning_rate}'
        log_dir = os.path.join(log_dir, run_dir)
        #hyper
        input_size =  2304 #48x48
        num_classes = 7
        num_epochs = 1
        writer = SummaryWriter(log_dir)


        train_dataset = readDataFCNN(Leaning_csv)
        test_dataset = readDataFCNN(test_csv_file)

        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   batch_size=batch_size,
                                                   shuffle=True)

        test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                   batch_size=batch_size,
                                                   shuffle=True)
        model = NeuralNet(input_size, num_classes)
        model.to(device)
        # loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        def validate(model, test_loader, epoch):
            with torch.no_grad():
                model.eval()
                n_correct = 0
                n_samples = 0
                for labels, images in test_loader:
                    images = images.to(device)
                    labels = labels.to(device)
                    outputs = model(images)

                    _, predictions = torch.max(outputs, 1)
                    n_samples += labels.shape[0]
                    n_correct += (predictions == labels).sum().item()

                acc = 100.0 * n_correct / n_samples
                print(f'Accuracy of the network on the test images: {acc} % in the batch size {batch_size} Learning rate {learning_rate}')
                return acc
                #writer.add_scalar('Validation Accuracy', acc, epoch)
                #writer.close()

            #traning
        n_total = len(train_loader)
        for epoch in range(num_epochs):
            model.train()
            n_correct = 0
            n_samples = 0
            for i, (labels, images) in enumerate(train_loader):
                images = images.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                # forward
                outputs = model(images)
                loss = criterion(outputs, labels.type(torch.int64))

                #backwards
                loss.backward()
                optimizer.step()

                #calculate traning accuracy
                _, predictions = torch.max(outputs, 1)
                n_samples += labels.shape[0]
                n_correct += (predictions == labels).sum().item()

                #if(i+1) % 100 == 0:
                #     print(f"epoch {epoch +1} / { num_epochs}, step {i+1}/{n_total}, loss: {loss}")

            acc = 100.0 * n_correct / n_samples

            print(f"Finish epoch {epoch + 1} of traning with tracc {acc} %")
            writer.add_scalar('Training Accuracy', acc, epoch)
            #test
            if epoch % 10 == 0 and epoch != 0:
                validate(model, test_loader,epoch)
        result[x][y] = validate(model, test_loader, epoch)
for i in range(len(result)):
    for j in range(len(result[i])):
        print(result[i][j])
# torch.save(model.state_dict(), 'feed-Forward.pth')
# validate(model,test_loader,99)


