import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from net import vgg16
img_pth='./data/input/image/057_jOjOWsmPwSs.jpg'
img=Image.open(img_pth)
'''处理图片'''
transform=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor()])
image=transform(img)
'''加载网络'''
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
net =vgg16()
model=torch.load("./Adogandcat.5.pth",map_location=device)
net.load_state_dict(model)
net.eval()
image=torch.reshape(image,(1,3,224,224))
with torch.no_grad():
    out=net(image)
out=F.softmax(out,dim=1)
out=out.data.cpu().numpy()
print(out)
a=int(out.argmax(1))
plt.figure()
list=["0",'1']
plt.suptitle("Classes:{}:{:.1%}".format(list[a],out[0,a]))
plt.imshow(img)
plt.show()