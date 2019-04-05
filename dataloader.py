import torch
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import transforms

__all__=["get_loader"]

train_t = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
test_t = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,),(0.3081,))
])

def get_loader(batch_size,num_workers=4,root_path="./data"):
    trainset = MNIST(root_path,True,transform=train_t,download=True)
    testset = MNIST(root_path,False,transform=test_t)

    trainloader = DataLoader(trainset,batch_size=batch_size,shuffle=True,num_workers=num_workers)
    testloader = DataLoader(testset,batch_size=batch_size,shuffle=False,num_workers=num_workers)
    return trainloader,testloader

if __name__=="__main__":
    trainloader , testloader = get_loader(32,2)
    for x,y in trainloader:
        print(x.size(),y.size())
        break
    for x,y in testloader:
        print(x.size(),y.size())
        break