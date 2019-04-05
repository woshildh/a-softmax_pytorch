import torch,os
import tensorboardX as tb
import matplotlib.pyplot as plt
import numpy as np
import imageio
from PIL import Image

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val):
        self.val = val
        self.sum += val
        self.count += 1
        self.avg = self.sum / self.count

class Logger(object):
    def __init__(self,tb_path="./logs/tblog/"):
        self.writer = tb.SummaryWriter(log_dir=tb_path)
    def log(self,step=1,content={}):
        for key,value in content.items():
            self.writer.add_scalar(key,value,step)

def save_checkpoints(name,epoch,state_dict,is_best=False,save_most=5):
    torch.save(state_dict,name.format(epoch))
    if is_best:
        torch.save(state_dict,name.format("best_acc"))
    if os.path.exists(name.format(epoch-save_most)):
        os.remove(name.format(epoch-save_most))

def plot_features(features, labels, num_classes, epoch, name):
    colors = ['C0','C1','C2','C3','C4','C5','C6','C7','C8','C9']
    for label_idx in range(num_classes):
        plt.scatter(
            features[labels==label_idx, 0],
            features[labels==label_idx, 1],
            c=colors[label_idx],
            s=1,
        )
    plt.legend(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'], loc='upper right')
    save_name = name.format(epoch)
    plt.savefig(save_name, bbox_inches='tight')
    plt.close()

def self_cmp(name):
    step = int(name.split("_")[-1].split(".")[0])
    return step

def make_gif(root_dir,save_path):
    file_list = os.listdir(root_dir)
    if len(file_list)==0:
        return
    file_list = sorted(file_list,key=self_cmp)
    image_list = []
    for file_ in file_list:
        image = imageio.imread(os.path.join(root_dir,file_))
        image = Image.fromarray(image).resize((500,500),Image.BILINEAR)
        image_list.append(image)
    imageio.mimsave(save_path,image_list,"GIF",duration=0.2)

if __name__=="__main__":
    # features = np.random.rand(10000,2)
    # labels = np.random.randint(low=0,high=10,size=(10000))
    # plot_features(features,labels,10,1,"./train_{}.png")
    make_gif("./logs/images/train/","./logs/train.gif")
