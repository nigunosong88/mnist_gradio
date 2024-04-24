import os
import time
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchmetrics import Accuracy
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision import datasets, transforms, models
import resnet18
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
import cv2 as cv
import gradio as gr

EPOCH = 50
BATCH_SIZE = 128
LR = 0.001

NUM_FEATURES = 28*28
NUM_CLASSES = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
"cuda" if torch.cuda.is_available() else "cpu"
def main():

    PATH_DATASETS = "data/" # 預設路徑
    # 下載 MNIST 手寫阿拉伯數字 訓練資料
    train_ds = MNIST(PATH_DATASETS, train=True, download=True, 
                    transform=transforms.ToTensor())

    # 下載測試資料
    test_ds = MNIST(PATH_DATASETS, train=False, download=True, 
                    transform=transforms.ToTensor())

    train_loader = DataLoader(dataset=train_ds, 
                            batch_size=BATCH_SIZE, 
                            shuffle=True)

    test_loader = DataLoader(dataset=test_ds, 
                            batch_size=BATCH_SIZE, 
                            shuffle=False)
    return train_loader,test_loader

    

def train_test(train_loader,test_loader):
    model = resnet18.ResNet18().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR) 

    start_time = time.time()
    for epoch in range(EPOCH):
        
        model.train()
        for batch_idx, (data, targets) in enumerate(train_loader):
            
            data = data.to(device)
            targets = targets.to(device)
                
            ### FORWARD AND BACK PROP
            output  = model(data)
            loss = criterion(output, targets)
            optimizer.zero_grad()
            
            loss.backward()
            
            ### UPDATE MODEL PARAMETERS
            optimizer.step()
            
            ### LOGGING
            if not batch_idx % 50:
                print ('Epoch: %03d/%03d | Batch %04d/%04d | Cost: %.4f' 
                    %(epoch+1, EPOCH, batch_idx, 
                        len(train_loader), loss))

            

        model.eval()
        correct = 0.0#正確
        Accuracy = 0.0#正确率
        text_loss = 0.0
        with torch.set_grad_enabled(False): # save memory during inference
            for data,target in (test_loader):
                data = data.to(device)
                target = target.to(device)
                output = model(data)#處理後的結果
                text_loss += F.cross_entropy(output,target).item()#計算測試損失之和
                pred = output.argmax(dim=1)#找到概率最大的索引
                correct += pred.eq(target.view_as(pred)).sum().item()#累计正确的次数
            text_loss /= len(test_loader.dataset)#損失和/数据集的總數量 = 平均loss
            Accuracy = 100.0*correct / len(test_loader.dataset)#正确个数/数据集的总数量 = 正确率
            print("Test__Average loss: {:4f},Accuracy: {:.3f}\n".format(text_loss,Accuracy))

        print('Time elapsed: %.2f min' % ((time.time() - start_time)/60))
        
    print('Total Training Time: %.2f min' % ((time.time() - start_time)/60))

    torch.save(model.state_dict(), 'resnet_model.pth')

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
    src = cv.imread(image_name+"background.jpg")#读取图片
    gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    ret, binary = cv.threshold(gray, 0, 255,cv.THRESH_BINARY_INV|cv.THRESH_OTSU)#白底黑字转换为黑底白字
    cv.imwrite(image_name+"block.jpg", binary)#将图像数据写入到图像文件中
    
def recognize_digit(img):
    image_name="num"
    image_name='data\\'+image_name
    #------------------------------------
    # 從字典中提取圖像數據
    img_data = img['composite']
    # 將sketchpad的輸入轉換為PIL圖像
    image = Image.fromarray(np.uint8(img_data))
    # 保存圖像
    image.save(image_name+".png")
    #------------------------------------
    # 使用示例
    add_white_background(image_name+".png", image_name+"background.jpg")
    imag(image_name+"background.jpg")
    #------------------------------------
    im = plt.imread(image_name+"block.jpg")  # 读入图片
    images = Image.open(image_name+"block.jpg")    # 将图片存储到images里面
    images = images.resize((28,28))   # 调整图片的大小为28*28
    images = images.convert('L')   # 灰度化
    #------------------------------------
    transform = transforms.ToTensor()#转换为tentor
    images = transform(images)#对图片进行transform
    images = images.resize(1,1,28,28)#调整图片尺寸（四维）
    # 加载网络和参数
    # model = ResNet18().to(device)#加载模型
    model = model.load_state_dict(torch.load('resnet_model.pth'))#加载参数
    model.eval()#测试模型
    images=images.to(device)
    outputs = model(images).to(device)#输出结果
    print(outputs)
    label = outputs.argmax(dim =1) # 返回最大概率值的下标
    print(label[0])
    plt.title('{}'.format(int(label)))
    plt.imshow(im)
    plt.show()
    #------------------------------------
    return label.item()
    




if __name__ == "__main__":
    train_loader,test_loader=main()
    train_test(train_loader,test_loader)

    iface=gr.Interface(
    fn=recognize_digit, 
    inputs="sketchpad", 
    outputs=["text"])

    #啟動界面
    iface.launch()