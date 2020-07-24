import cv2
import numpy as np
import os
from PIL import Image, ImageEnhance, ImageFilter
import time
from multiprocessing import Pool
def get_noise(value=10, size=(32, 32, 3)):   
    noise = np.random.uniform(0, 256, size)
#     noise = [noise,noise,noise]
#     noise = np.array(noise)
    #控制噪声水平，取浮点数，只保留最大的一部分作为噪声
    v = value *0.03
    noise[np.where(noise<(256-v))]=0
        
#     #噪声做初次模糊
    k = np.array([ [0, 0.1, 0],
                    [0.1,  8, 0.1],
                    [0, 0.1, 0] ])
            
    noise = cv2.filter2D(noise,-1,k)   
    #可以输出噪声看看
    return noise

def rain_blur(noise, length=20, angle=135,w=1):
    '''
    将噪声加上运动模糊,模仿雨滴
    
    >>>输入
    noise：输入噪声图，shape = img.shape[0:2]
    length: 对角矩阵大小，表示雨滴的长度
    angle： 倾斜的角度，逆时针为正
    w:      雨滴大小
    
    >>>输出带模糊的噪声
    
    '''
    
    
    #这里由于对角阵自带45度的倾斜，逆时针为正，所以加了-45度的误差，保证开始为正
    trans = cv2.getRotationMatrix2D((length/2, length/2), angle-45, 1-length/100.0)  
    dig = np.identity(length)   #生成对焦矩阵
    k = cv2.warpAffine(dig, trans, (length, length))  #生成模糊核
    k = cv2.GaussianBlur(k,(w,w),0)    #高斯模糊这个旋转后的对角核，使得雨有宽度
    
    #k = k / length                         #是否归一化
    
    blurred = cv2.filter2D(noise, -1, k)    #用刚刚得到的旋转后的核，进行滤波
    
    #转换到0-255区间
    cv2.normalize(blurred, blurred, 0, 255, cv2.NORM_MINMAX)
    blurred = np.array(blurred, dtype=np.uint8)
    
    return blurred

def alpha_rain(rain,img,beta = 0.8):
    
    #输入雨滴噪声和图像
    #beta = 0.8   #results weight
    #显示下雨效果
    
#     rain = np.array(rain,dtype=np.float32)     #数据类型变为浮点数，后面要叠加，防止数组越界要用32位
#     rain_result[:,:,0]= rain_result[:,:,0] * (255-rain[:,:,0])/255.0 + beta*rain[:,:,0]
#     rain_result[:,:,1] = rain_result[:,:,1] * (255-rain[:,:,0])/255 + beta*rain[:,:,0] 
#     rain_result[:,:,2] = rain_result[:,:,2] * (255-rain[:,:,0])/255 + beta*rain[:,:,0]
    #对每个通道先保留雨滴噪声图对应的黑色（透明）部分，再叠加白色的雨滴噪声部分（有比例因子）
    img = np.array(img, dtype = np.float32)
    alpha = (255 - rain)/255
    img = np.multiply(img, alpha) + beta * rain
    img = img.astype(np.uint8)
    return Image.fromarray(img)

def resize_img_folder(in_dir,folder, out_dir):
    print('Folder %s' % in_dir)
    out_dir=os.path.join(out_dir,folder)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    for filename in os.listdir(in_dir):
        # Exception raised when file is not an image
        im = Image.open(os.path.join(in_dir, filename))
        # Convert grayscale images into 3 channels
        if im.mode != "RGB":
            im = im.convert(mode="RGB")
            # localization
            # blur bigger
        im_resized = im.filter(ImageFilter.BoxBlur(0.05))
        # darken smaller
        im_resized = ImageEnhance.Brightness(im_resized).enhance(0.3)
        # rain drops
        blurred = rain_blur(get_noise(value=1, size=np.shape(im)), length=20, w=9)
        im_resized = alpha_rain(blurred, im_resized, beta=0.4)
        # Get rid of extension (.jpg or other)
        filename = os.path.splitext(filename)[0]
        im_resized.save(os.path.join(out_dir, filename + '.JPEG'))

if __name__ == "__main__":
    start=time.time()
    in_dir='/raid/data/guozg/data_train_32/'
    current_out_dir='/raid/data/wangb/imagenet32/train/'
    # in_dir='/home/guozg/imagenet32/data/'
    # current_out_dir='/raid/data/wangb/imagenet32/val/'
    folders = [dir for dir in sorted(os.listdir(in_dir)) if os.path.isdir(os.path.join(in_dir, dir))]
    p=Pool(8)
    for i, folder in enumerate(folders):
        ifolder=os.path.join(in_dir,folder)
        p.apply_async(resize_img_folder, args=(ifolder, folder, current_out_dir))
    p.close()
    p.join()
    print("ok")
    print(f"耗时：{time.time() - start}S")
