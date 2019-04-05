import torch
import models,a_softmax,utils,dataloader
import torch.nn.functional as F
import tensorboardX
import numpy as np
import torch.optim.lr_scheduler as lr_sche

#设定一些超参数
batch_size=64
epochs=100
classes_num=10
model_lr=0.001
use_gpu=True
display_step=40

def train(net,optimizer,criterion,trainloader,epoch):
    loss_log = utils.AverageMeter()
    acc = utils.AverageMeter()

    #用于收集所有的特征和标签
    all_feature,all_labels=[],[]
    for i,(x,y) in enumerate(trainloader):
        if use_gpu:
            x , y=x.cuda() , y.cuda()
        y_,feature = net(x)
        #计算损失
        loss = criterion(y_,y)
        #梯度清0
        optimizer.zero_grad()
        #损失更新
        loss.backward()
        #模型更新
        optimizer.step()
        #计算正确率
        predicts = torch.argmax(y_[0],dim=1,keepdim=False)
        correct_num = (predicts==y).sum().item()
        step_acc = correct_num/x.size(0)
        #记录这些值
        loss_log.update(loss.item())
        acc.update(step_acc)
        #收集结果
        all_feature.append(feature.detach().cpu().numpy())
        all_labels.append(y.detach().cpu().numpy())
        #输出结果
        if i % display_step==0:
            print("{} epoch, {} step,step acc is {:.4f}, step loss is {:.5f}".format(
                epoch,i,step_acc,loss.item(),
            ))
    #将收集的特征和label合并
    all_feature = np.concatenate(all_feature,axis=0)
    all_labels = np.concatenate(all_labels,axis=0)
    #返回结果
    return acc.avg,loss_log.avg,all_feature,all_labels

def test(net,criterion,testloader,epoch):
    acc = utils.AverageMeter()
    loss_log = utils.AverageMeter()

    all_feature,all_labels=[],[]
    #用于收集所有的特征和标签
    all_feature,all_labels=[],[]
    for i,(x,y) in enumerate(testloader):
        if use_gpu:
            x , y = x.cuda() , y.cuda()
        y_,feature = net(x)
        #计算损失
        loss = criterion(y_,y).item()
        #计算正确率
        predicts = torch.argmax(y_[0],dim=1,keepdim=False)
        correct_num = (predicts==y).sum().item()
        step_acc = correct_num/x.size(0)
        #记录这些值
        acc.update(step_acc)
        loss_log.update(loss)
        #收集结果
        all_feature.append(feature.detach().cpu().numpy())
        all_labels.append(y.detach().cpu().numpy())
    #将收集的特征和label合并
    all_feature = np.concatenate(all_feature,axis=0)
    all_labels = np.concatenate(all_labels,axis=0)
    #返回结果
    return acc.avg,loss_log.avg,all_feature,all_labels

def main():
    #定义网络
    net = models.LeNetWithAngle(classes_num)
    if use_gpu:
        net=net.cuda()
    #定义优化器
    optimizer = torch.optim.SGD(net.parameters(),lr=model_lr,weight_decay=1e-5,
        nesterov=True,momentum=0.9)
    print("net and optimzer load succeed")
    #定义数据加载
    trainloader , testloader = dataloader.get_loader(batch_size=batch_size,
         root_path="./data/MNIST")
    print("data load succeed")
    #定义logger
    logger = utils.Logger(tb_path="./logs/tblog/")
    #定义学习率调整器
    scheduler = lr_sche.StepLR(optimizer,30,0.1)
    #定义损失函数
    criterion = a_softmax.AngleSoftmaxLoss(gamma=0)
    best_acc=0
    #开始训练
    for i in range(1,epochs+1):
        scheduler.step(epoch=i)
        net.train()
        train_acc,train_loss,all_feature,all_labels=\
            train(net,optimizer,criterion,trainloader,i)
        utils.plot_features(all_feature,all_labels,classes_num,i,"./logs/images/train/train_{}.png")
        net.eval()
        test_acc,test_loss,all_feature,all_labels=test(net,criterion,testloader,i)
        utils.plot_features(all_feature,all_labels,classes_num,i,"./logs/images/test/test_{}.png")
        print("{} epoch end, train acc is {:.4f}, test acc is {:.4f}".format(
            i,train_acc,test_acc))
        content={"Train/acc":train_acc,"Test/acc":test_acc,
            "Train/loss":train_loss,"Test/loss":test_loss
        }
        logger.log(step=i,content=content)
        if best_acc < test_acc:
            best_acc=test_acc
        utils.save_checkpoints("./logs/weights/net_{}.pth",i,\
            net.state_dict(),(best_acc==test_acc))
    utils.make_gif("./logs/images/train/","./logs/train.gif")
    utils.make_gif("./logs/images/test/","./logs/test.gif")
    print("Traing finished...")

if __name__=="__main__":
    main()
