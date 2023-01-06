import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from torch import nn
import torch.optim as optim
#from torchvision.transforms.functional import to_pil_image
import matplotlib.pyplot as plt

def image_preprocess(img_dir):
    img = Image.open(img_dir)
    transform = transforms.Compose([
                    transforms.Resize(image_size),
                    transforms.CenterCrop(image_size),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.40760392, 0.45795686, 0.48501961], 
                                        std=[1,1,1]),
                    #transforms.Normalize([0.5], [0.5])
                ])
    img = transform(img).view((-1,3,image_size,image_size))
    return img

def image_postprocess(tensor):
    transform = transforms.Normalize(mean=[-0.40760392, -0.45795686, -0.48501961], 
                                     std=[1,1,1])
    #transform = transforms.Normalize([0.5], [0.5])
    img = transform(tensor.clone())
    img = img.clamp(0,1)
    img = torch.transpose(img,0,1)
    img = torch.transpose(img,1,2)
    return img

class Resnet(nn.Module):
    def __init__(self):
        super(Resnet,self).__init__()
        self.layer0 = nn.Sequential(*list(resnet.children())[0:1])
        self.layer1 = nn.Sequential(*list(resnet.children())[1:4])
        self.layer2 = nn.Sequential(*list(resnet.children())[4:5])
        self.layer3 = nn.Sequential(*list(resnet.children())[5:6])
        self.layer4 = nn.Sequential(*list(resnet.children())[6:7])
        self.layer5 = nn.Sequential(*list(resnet.children())[7:8])

    def forward(self,x):
        out_0 = self.layer0(x)
        out_1 = self.layer1(out_0)
        out_2 = self.layer2(out_1)
        out_3 = self.layer3(out_2)
        out_4 = self.layer4(out_3)
        out_5 = self.layer5(out_4)
        return out_0, out_1, out_2, out_3, out_4, out_5

class GramMatrix(nn.Module):
    def forward(self, input):
        b,c,h,w = input.size()
        F = input.view(b, c, h*w)
        G = torch.bmm(F, F.transpose(1,2)) 
        return G

class GramMSELoss(nn.Module):
        def forward(self, input, target):
            out = nn.MSELoss()(GramMatrix()(input), target)
            return out
if __name__ == '__main__':
    resnet = models.resnet50(pretrained=True)
    content_dir = "C:/Users/HP/Desktop/python/style_transfer/input.jpg"
    style_dir = "C:/Users/HP/Desktop/python/style_transfer/style_img2.png"
    image_size=512
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #resnet = Resnet().to(device)
    #resnet.load_state_dict(torch.load('model.pth'))
    resnet = torch.load('model.pth').to(device)
    resnet.eval()
    for param in resnet.parameters():
        param.requires_grad = False    

    content = image_preprocess(content_dir).to(device)
    style = image_preprocess(style_dir).to(device)
    generated = content.clone().requires_grad_().to(device)

    print(content.requires_grad,style.requires_grad,generated.requires_grad)

    gen_img = image_postprocess(generated[0].cpu()).data.numpy()
    optimizer = optim.LBFGS([generated])

    style_target = list(GramMatrix().to(device)(i) for i in resnet(style))
    content_target = resnet(content)[1]
    style_weight = [1/n**2 for n in [64,64,256,512,1024,2048]]

    out = resnet(generated)

    gen_img = image_postprocess(generated[0].cpu()).data.numpy()
    print(gen_img.shape)
    kk = Image.fromarray(gen_img)
    kk.save('result.png')