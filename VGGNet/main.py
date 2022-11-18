import torch
import torch.nn as nn
from net import vgg16
from  torch.utils.data import DataLoader
from data import *
'''数据集'''
annotation_path='cls_train.txt'
with open(annotation_path,'r') as f:
    lines=f.readlines()
np.random.seed(10101)
np.random.shuffle(lines)#打乱数据
np.random.seed(None)
num_val=int(len(lines)*0.1)
num_train=len(lines)-num_val
#输入图像大小
input_shape=[224,224]
train_data=DataGenerator(lines[:num_train],input_shape,True)
val_data=DataGenerator(lines[num_train:],input_shape,False)
val_len=len(val_data)
"""加载数据"""
gen_train=DataLoader(train_data,batch_size=4)
gen_test=DataLoader(val_data,batch_size=4)
'''构建网络'''
device=torch.device('cuda'if torch.cuda.is_available() else "cpu")
net=vgg16(pretrained=True, progress=True,num_classes=2)
net.to(device)
'''选择优化器和学习率的调整方法'''
lr=0.0001
optim=torch.optim.Adam(net.parameters(),lr=lr)
sculer=torch.optim.lr_scheduler.StepLR(optim,step_size=1)
'''训练'''
epochs=50
for epoch in range(epochs):
    sculer.step()
    total_train=0
    for data in gen_train:
        img,label=data
        with torch.no_grad():
            img =img.to(device)
            label=label.to(device)
        optim.zero_grad()
        output=net(img)
        train_loss=nn.CrossEntropyLoss()(output,label).to(device)
        train_loss.backward()
        optim.step()
        total_train+=train_loss
    total_test=0
    total_accuracy=0
    for data in gen_test:
        img,label =data
        with torch.no_grad():
            img=img.to(device)
            label=label.to(device)
            optim.zero_grad()
            out=net(img)
            test_loss=nn.CrossEntropyLoss()(out,label).to(device)
            total_test+=test_loss
            accuracy=(out.argmax(1)==label).sum()
            total_accuracy+=accuracy
    print("训练集上的损失：{}".format(total_train))
    print("测试集上的损失：{}".format(total_test))
    print("测试集上的精度：{:.1%}".format(total_accuracy/val_len))
    # torch.save(net,"dogandcat.{}.pt".format(epoch+1))
    torch.save(net.state_dict(),"Adogandcat.{}.pth".format(epoch+1))
    print("模型已保存")


