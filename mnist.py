import gradio as gr
import torch
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from torchvision import transforms
from torch import nn
from torch.nn import functional as F
import cv2 as cv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
"cuda" if torch.cuda.is_available() else "cpu"
EPOCH = 50
BATCH_SIZE = 128
LR = 0.001

GRAYSCALE = True
NUM_FEATURES = 28*28
NUM_CLASSES = 10
class ResBlk(nn.Module):  # 定义Resnet Block模块
    """
    resnet block
    """

    def __init__(self, ch_in, ch_out, stride=1):  # 进入网络前先得知道传入层数和传出层数的设定
        """
        :param ch_in:
        :param ch_out:
        """
        super(ResBlk, self).__init__()  # 初始化

        # we add stride support for resbok, which is distinct from tutorials.
        # 根据resnet网络结构构建2个（block）块结构 第一层卷积 卷积核大小3*3,步长为1，边缘加1
        self.conv1 = nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=stride, padding=1)
        # 将第一层卷积处理的信息通过BatchNorm2d
        self.bn1 = nn.BatchNorm2d(ch_out)
        # 第二块卷积接收第一块的输出，操作一样
        self.conv2 = nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(ch_out)

        # 确保输入维度等于输出维度
        self.extra = nn.Sequential()  # 先建一个空的extra
        if ch_out != ch_in:
            # [b, ch_in, h, w] => [b, ch_out, h, w]
            self.extra = nn.Sequential(
                nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=stride),
                nn.BatchNorm2d(ch_out)
            )

    def forward(self, x):  # 定义局部向前传播函数
        """
        :param x: [b, ch, h, w]
        :return:
        """
        out = F.relu(self.bn1(self.conv1(x)))  # 对第一块卷积后的数据再经过relu操作
        out = self.bn2(self.conv2(out))  # 第二块卷积后的数据输出
        # short cut.
        # extra module: [b, ch_in, h, w] => [b, ch_out, h, w]
        # element-wise add:
        out = self.extra(x) + out  # 将x传入extra经过2块（block）输出后与原始值进行相加
        out = F.relu(out)  # 调用relu，这里使用F.调用

        return out


class ResNet18(nn.Module):  # 构建resnet18层

    def __init__(self):
        super(ResNet18, self).__init__()

        self.conv1 = nn.Sequential(  # 首先定义一个卷积层
            nn.Conv2d(1, 32, kernel_size=3, stride=3, padding=0),
            nn.BatchNorm2d(32)
        )
        # followed 4 blocks 调用4次resnet网络结构，输出都是输入的2倍
        # [b, 64, h, w] => [b, 128, h ,w]
        self.blk1 = ResBlk(32, 64, stride=1)
        # [b, 128, h, w] => [b, 256, h, w]
        self.blk2 = ResBlk(64, 128, stride=1)
        # # [b, 256, h, w] => [b, 512, h, w]
        self.blk3 = ResBlk(128, 256, stride=1)
        # # [b, 512, h, w] => [b, 1024, h, w]
        self.blk4 = ResBlk(256, 256, stride=1)

        self.outlayer = nn.Linear(256 * 1 * 1, 10)  # 最后是全连接层

    def forward(self, x):  # 定义整个向前传播
        """
        :param x:
        :return:
        """
        x = F.relu(self.conv1(x))  # 先经过第一层卷积

        # [b, 64, h, w] => [b, 1024, h, w]
        x = self.blk1(x)  # 然后通过4次resnet网络结构
        x = self.blk2(x)
        x = self.blk3(x)
        x = self.blk4(x)

        # print('after conv:', x.shape) #[b, 512, 2, 2]
        # F.adaptive_avg_pool2d功能尾巴变为1,1，[b, 512, h, w] => [b, 512, 1, 1]
        x = F.adaptive_avg_pool2d(x, [1, 1])
        # print('after pool:', x.shape)
        x = x.view(x.size(0), -1)  # 平铺一维值
        x = self.outlayer(x)  # 全连接层

        return x
model = ResNet18().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR) 

im = plt.imread('renum\\7.jpg')  # 读入图片
images = Image.open('renum\\7.jpg')    # 将图片存储到images里面
images = images.resize((28,28))   # 调整图片的大小为28*28
images = images.convert('L')   # 灰度化

transform = transforms.ToTensor()#转换为tentor
images = transform(images)#对图片进行transform
images = images.resize(1,1,28,28)#调整图片尺寸（四维）
# 加载网络和参数
# model = ResNet18().to(device)#加载模型
model.load_state_dict(torch.load('resnet_model.pth'))#加载参数
model.eval()#测试模型
images=images.to(device)
outputs = model(images).to(device)#输出结果
label = outputs.argmax(dim =1) # 返回最大概率值的下标

def add_white_background(input_path, output_path):
    # 打开原始图像
    original_image = Image.open(input_path)

    # 创建白色背景图像，与原始图像相同大小和模式
    white_background = Image.new("RGB", original_image.size, (255, 255, 255))

    # 将原始图像粘贴到白色背景上
    white_background.paste(original_image, (0, 0), original_image)

    # 保存结果图像
    white_background.save(output_path)

def imag(image_name):# 调整图片大小
    src = cv.imread('new\\'+image_name+"1.jpg")#读取图片
    gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    ret, binary = cv.threshold(gray, 0, 255,cv.THRESH_BINARY_INV|cv.THRESH_OTSU)#白底黑字转换为黑底白字
    cv.imwrite('new\\'+image_name+"2.jpg", binary)#将图像数据写入到图像文件中
    

def recognize_digit(img):
  image_name="123"
  #------------------------------------
  # 從字典中提取圖像數據
  img_data = img['composite']
  # 將sketchpad的輸入轉換為PIL圖像
  image = Image.fromarray(np.uint8(img_data))
  # 保存圖像
  image.save('new\\'+image_name+".png")
  #------------------------------------
  # 使用示例
  add_white_background('new\\'+image_name+".png", 'new\\'+image_name+"1.jpg")
  imag(image_name)
  #------------------------------------
  im = plt.imread('new\\'+image_name+"2.jpg")  # 读入图片
  images = Image.open('new\\'+image_name+"2.jpg")    # 将图片存储到images里面
  images = images.resize((28,28))   # 调整图片的大小为28*28
  images = images.convert('L')   # 灰度化
  #------------------------------------
  transform = transforms.ToTensor()#转换为tentor
  images = transform(images)#对图片进行transform
  images = images.resize(1,1,28,28)#调整图片尺寸（四维）
  # 加载网络和参数
  # model = ResNet18().to(device)#加载模型
  model.load_state_dict(torch.load('resnet_model.pth'))#加载参数
  model.eval()#测试模型
  images=images.to(device)
  outputs = model(images).to(device)#输出结果
#   print(outputs)
  label = outputs.argmax(dim =1) # 返回最大概率值的下标
#   print(label[0])
#   plt.title('{}'.format(int(label)))
#   plt.imshow(im)
#   plt.show()
  #------------------------------------
  return label.item()

iface=gr.Interface(
    fn=recognize_digit, 
    inputs="sketchpad", 
    outputs=["text"])

#啟動界面
iface.launch()