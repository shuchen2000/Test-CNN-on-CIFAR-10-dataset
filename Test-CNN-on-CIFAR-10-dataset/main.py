import copy
import torch
import torch.nn.functional as fun
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

#the final accuracy is about 69%

gpu_index = torch.cuda.current_device()
torch.cuda.set_device(gpu_index)

BATCH_SIZE_TRAIN = 4
BATCH_SIZE_TEST = 4

def get_percentage_str(a):
    return '%f %%' % (100.00 * a)

def gaussian_weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.04)

transform = transforms.Compose([transforms.ToTensor()])

trainDataset = torchvision.datasets.CIFAR10(root='.\data' , train=True , download=True , transform=transform)
trainLoader = torch.utils.data.DataLoader(trainDataset , batch_size=BATCH_SIZE_TRAIN , shuffle=True , num_workers=0)

testDataset = torchvision.datasets.CIFAR10(root='.\data' , train=False , download=True , transform=transform)
testLoader = torch.utils.data.DataLoader(testDataset , batch_size=BATCH_SIZE_TEST , shuffle=False , num_workers=0)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

class CNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        #[32,32,3]
        self.conv1 = torch.nn.Conv2d(in_channels=3 , out_channels=128 , kernel_size=7 , stride=1 , padding=1).apply(gaussian_weights_init)
        #[28,28,128]
        self.norm1 = torch.nn.BatchNorm2d(num_features=128)
        #[28,28,128]
        self.pool1 = torch.nn.MaxPool2d(kernel_size=2,stride=2)
        #[14,14,128]
        self.conv2 = torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5, stride=1, padding=1).apply(gaussian_weights_init)
        #[12,12,192]
        self.norm2 = torch.nn.BatchNorm2d(num_features=256)
        # [12,12,192]
        self.pool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        # [6*6*192]
        self.conv3 = torch.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=0).apply(gaussian_weights_init)
        # [4,4,256]
        self.norm3 = torch.nn.BatchNorm2d(num_features=512)
        # [4,4,256]
        self.pool3 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        # [2*2*256]
        self.dp = torch.nn.Dropout(0.2)
        self.fc1 = torch.nn.Linear(512*2*2,512)
        self.fc2 = torch.nn.Linear(512, 256)
        self.fc3 = torch.nn.Linear(256, 64)
        self.fc4 = torch.nn.Linear(64, 10)

    def forward(self,x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = fun.relu(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.norm2(x)
        x = fun.relu(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.norm3(x)
        x = fun.relu(x)
        x = self.pool3(x)

        x = x.view(-1 , 2*2*512)

        x = self.dp(x)
        x = self.fc1(x)
        x = fun.relu(x)

        x = self.dp(x)
        x = self.fc2(x)
        x = fun.relu(x)

        x = self.fc3(x)
        x = fun.relu(x)

        x = self.fc4(x)
        return x

def train(path):
    accuracyTemp = 0.0
    epoch = 0
    model = CNN().cuda()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    loss_func = torch.nn.CrossEntropyLoss()
    while (True):
        model.train()
        times = 0
        running_loss = 0
        for step,data in enumerate(trainLoader):
            times = times+1
            b_x, b_y = data
            b_x = b_x.cuda()
            b_y = b_y.cuda()
            optimizer.zero_grad()
            outputs = model.forward(b_x)
            loss = loss_func(outputs, b_y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if times % 100 == 99:  #
                print('[%d, %8d] loss: %.3f' %
                      (epoch + 1, times + 1, running_loss / 2000))
                running_loss = 0.0
        model.eval()
        accuracyCurrent = val(model)
        if (accuracyTemp > accuracyCurrent):
            print('This epoch of training reduces the accuracy of the model on the test set (' + get_percentage_str(accuracyTemp) + '->' + get_percentage_str(accuracyCurrent) + ')' + ' , the last model is saved . STOP')
            break
        else:
            print('This epoch of training improves the accuracy of the model on the test set ('+get_percentage_str(accuracyTemp)+'->'+get_percentage_str(accuracyCurrent)+')'+' ,  the current model is saved . CONTINUE')
            accuracyTemp = accuracyCurrent
            torch.save(model, path)
        epoch = epoch+1
    return torch.load(r'.\model\CNN.pth')


def val(model):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for step,data in enumerate(testLoader):
            b_x, b_y = data
            b_x = b_x.cuda()
            b_y = b_y.cuda()
            outputs = model.forward(b_x)
            numbers, predicted = torch.max(outputs.data, 1)
            total += b_y.size(0)
            correct += (predicted == b_y).sum().item()
    return float(correct) / float(total)

def testWithId(model,i):
    i=int(i)
    b_x, b_y = testDataset[i]
    mat = np.rot90(b_x.T, k=3, axes=(0, 1))
    plt.imshow(mat)
    outputs = model.forward(b_x.cuda().view(1, 3, 32, 32)).cpu()
    str = "label:  " + classes[np.array(b_y)] + "\n" + "predc:  " + classes[np.array(torch.max(outputs.data, 1).indices[0])] + "\n"
    if (np.array(b_y) == np.array(torch.max(outputs.data, 1).indices)):
        str = str + "CORRECT"
    else:
        str = str + "WRONG"
    plt.title(str)
    plt.show()


model = train(r'.\model\CNN.pth')
model.eval()
print(val(model))



