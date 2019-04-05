import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from torch.autograd import Variable

__all__=["AngleLinear","AngleSoftmaxLoss"]

class AngleLinear(nn.Module):
    def __init__(self,in_planes,out_planes,m=4):
        super(AngleLinear,self).__init__()  
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.m = m
        self.weight = nn.Parameter(torch.Tensor(in_planes,out_planes))
        self.weight.data.uniform_(-1,1).renorm_(2,1,1e-5).mul_(1e5)
        self.cos_function=[
            lambda x: x**0,
            lambda x: x**1,
            lambda x: 2*x**2-1,
            lambda x: 4*x**3-3*x,
            lambda x: 8*x**4-8*x**2+1,
            lambda x: 16*x**5-20*x**3+5*x,
        ]
    def forward(self,x):
        '''
        inputs:
            x: [batch,in_planes]
        return:
            cos_x: [batch,out_planes]
            phi_x: [batch,out_planes]
        '''
        w = self.weight.renorm(2, 1, 1e-5).mul(1e5) #[batch,out_planes]
        x_modulus = x.pow(2).sum(1).pow(0.5) #[batch]
        w_modulus = w.pow(2).sum(0).pow(0.5) #[out_planes]
        # get w@x=||w||*||x||*cos(theta)
        inner_wx = x.mm(w) # [batch,out_planes]
        cos_theta = (inner_wx/x_modulus.view(-1,1))/w_modulus.view(1,-1)
        cos_theta = cos_theta.clamp(-1,1)
        # get cos(m*theta)
        cos_m_theta = self.cos_function[self.m](cos_theta)
        theta = Variable(cos_theta.data.acos())
        #get k, theta is in [ k*pi/m , (k+1)*pi/m ]
        k = (self.m * theta / math.pi).floor()
        minus_one = k*0 - 1
        # get phi_theta = -1^k*cos(m*theta)-2*k
        phi_theta = (minus_one ** k) * cos_m_theta - 2 * k
        # get cos_x and phi_x
        # cos_x = cos(theta)*||x||
        # phi_x = phi(theta)*||x||
        cos_x = cos_theta * x_modulus.view(-1,1)
        phi_x = phi_theta * x_modulus.view(-1,1)
        return cos_x , phi_x

class AngleSoftmaxLoss(nn.Module):
    def __init__(self,gamma=0):
        super(AngleSoftmaxLoss, self).__init__()
        self.gamma   = gamma
        self.it = 0
        self.LambdaMin = 5.0
        self.LambdaMax = 1500.0
        self.lamb = 1500.0
    def forward(self,inputs,target):
        '''
        inputs:
            cos_x: [batch,classes_num]
            phi_x: [batch,classes_num]
            target: LongTensor,[batch]
        return:
            loss:scalar
        '''
        self.it += 1
        cos_x,phi_x = inputs
        target = target.view(-1,1)
        # get one_hot mat
        index = cos_x.data * 0.0 #size=(B,Classnum)
        index.scatter_(1,target.data.view(-1,1),1)
        index = index.byte()
        index = Variable(index)
        # set lamb, change the rate of softmax and A-softmax 
        self.lamb = max(self.LambdaMin,self.LambdaMax/(1+0.1*self.it))
        # get a-softmax and softmax mat
        output = cos_x * 1
        output[index] -= (cos_x[index] * 1.0/(+self.lamb))
        output[index] += (phi_x[index] * 1.0/(self.lamb))
        # get loss
        logpt = F.log_softmax(output,dim=1) #[batch,classes_num]
        logpt = logpt.gather(1,target) #[batch]
        pt = logpt.data.exp()
        loss = -1 * logpt * (1-pt)**self.gamma
        loss = loss.mean()
        return loss

if __name__=="__main__":
    layer = AngleLinear(2,5,m=3)
    x = torch.randn(8,2)
    target = torch.Tensor([0,1,2,4,0,1,3,2]).long()
    cos_x,phi_x = layer(x)
    criterion = AngleSoftmaxLoss()
    loss = criterion((cos_x,phi_x),target)
    print(loss)
