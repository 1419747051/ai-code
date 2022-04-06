import argparse
import cv2
import numpy as np


#设置参数
ap = argparse.ArgumentParser()
ap.add_argument("-i","--image",required=True,
                help="Path to the image to be scanned")

args = vars(ap.parse_args())

def order_points(pts):
    # 一个4个坐标
    rect = np.zeros((4,2),dtype="float32")

    #按顺序找到对应坐标0123分别是左上，左下，右上右下
    #计算左上，左下
    s=pts.sum(axis=1)
    rect[0] =pts[np.argmin(s)]
    rect[2] =pts[np.argmax(s)]

    #计算右上右下
    diff = np.diff(pts,axis =1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect

def four_point_transform(image,pts):
    # 获取输入坐标点
    rect = order_points(pts)
    (tl,tr,br,bl)=rect

    #计算输入的w和h的值
    widthA =np.sqrt(((br[0]-bl[0])**2)+((br[1]-bl[1])**2))
    widthB =np.sqrt(((tr[0]-tl[0])**2)+((tr[1]-tl[1])**2))
    maxWidth = max(int(widthA),int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    #变换后对应的坐标位置
    dst= np.array([
        [0,0],
        [maxWidth-1,0],
        [maxWidth-1,maxHeight-1],
        [0,maxHeight-1]],dtype="float32")

    #计算变换矩阵
    M= cv2.getPerspectiveTransform(rect,dst)
    warped = cv2.warpPerspective(image,M,(maxWidth,maxHeight))

    #返回变换后的结果
    return warped

def resize(image, width=None,height=None,inter=cv2.INTER_AREA):
    dim =None
    (h,w) = image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        r = height /float(h)
        dim = (int(w*r),height)
    else:
        r = width/float(w)
        dim = (width,int(h*r))
    resized = cv2.resize(image,dim,interpolation=inter)
    return  resized

# 读取输入
image = cv2.imread(args["image"])
#坐标也会相同变化
ratio = image.shape[0]/500.0
orig =image.copy()

image= resize(orig,height= 500)

#预处理
gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
gray = cv2.GaussianBlur(gray,(5,5),0)
edged = cv2.Canny(gray,75,200)

#显示预处理结果
print("setp 1:边缘检测")
cv2.imshow("image",image)
cv2.imshow("edged",edged)
cv2.waitKey(0)
cv2.destroyAllWindows()

#轮廓检测
cnts = cv2.findContours(edged.copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)[0]
cnts =sorted(cnts,key=cv2.contourArea,reverse=True)[:5]

for c in cnts:
    # 计算轮廓近似
    peri = cv2.arcLength(c,True)
    #c表示输入的点集
    #epsilon表示从原始轮廓到近似轮廓的最大距离，他说一个准确度参数
    #Ture表示封闭的
    approx =cv2.approxPolyDP(c,0.02*peri,True)

    #4个点的时候就拿出来
    if len(approx)==4:
        screenCnt = approx
        break

#展示结果
print("step2:获取轮廓")
cv2.drawContours(image,[screenCnt],-1,(0,255,0),2)
cv2.imshow("outline",image)
cv2.waitKey(0)
cv2.destroyAllWindows()

#透视变换
warped = four_point_transform(orig,screenCnt.reshape(4,2)*ratio)

#二值处理
warped = cv2.cvtColor(warped,cv2.COLOR_BGR2GRAY)
ref = cv2.threshold(warped,100,255,cv2.THRESH_BINARY)[1]
cv2.imwrite('scan1.jpg',ref)

#展示结果
print("step3: 变换")
cv2.imshow("original",resize(orig,height=650))
cv2.imshow("scanned",resize(ref,height=650))
cv2.waitKey(0)
