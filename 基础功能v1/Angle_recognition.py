import time
import cv2
import numpy as np
from PIL import Image
import math
import xlwt
import xlrd

cap = cv2.VideoCapture(0)    #打开摄像头


def recognition(file):
    # 读取照片
    img = cv2.imread(file, 3)

    # 高斯双边滤波，局部清晰

    img = cv2.bilateralFilter(img, 0, 50, 10)

    # RGB-RGB2HSV

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    image1 = Image.fromarray(hsv.astype('uint8'))

    # 去除背景

    lower_blue = np.array([10, 43, 46])
    upper_blue = np.array([100, 255, 255])

    lower_yellow = np.array([15, 50, 50])
    upper_yellow = np.array([35, 255, 255])
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)  # 设阈值，去除背景部分，留下黄色部分

    img_bg = cv2.bitwise_and(img, img, mask=mask)  # BGR #按位与--带掩码
    ##mask起掩码作用，当mask像素不为0时，做正常与操作，当mask像素为0时直接做0处理
    # 【mask为黑白图像时：纯白色部分进行正常的按位操作,mask为非纯白色部分设置为0即黑色】

    # BGR->RGB
    image1 = cv2.cvtColor(img_bg, cv2.COLOR_BGR2RGB)
    kernel = np.ones((3, 3), np.uint8)

    # 膨胀处理

    o = cv2.dilate(image1, kernel, iterations=2)  # 膨胀处处理       #
    o = cv2.medianBlur(o, 5)
    # 边缘检测

    binary = cv2.Canny(o, 500, 500)  # Canny边缘检测是一种流行的边缘检测算法   后两个值越大，非边缘点越少        #

    lines = cv2.HoughLinesP(binary, 1, np.pi / 180, 15, minLineLength=40, maxLineGap=10)  # 直线检测                #

    xs1 = [0 for i in range(100)]
    ys1 = [0 for i in range(100)]
    xs2 = [0 for i in range(100)]
    ys2 = [0 for i in range(100)]
    dig = [0 for i in range(100)]

    # 角度计算

    i = 0
    for line in lines:      #画面中没有黄色部分或者没有直线时易出错
        x1, y1, x2, y2 = line[0]
        xs1[i] = x1
        xs2[i] = x2
        ys1[i] = y1
        ys2[i] = y2
        # cv2.line(img, (x1, y1),(x2,y2), (0, 0, 255),2 )
        i = i + 1
    #    if  (abs(x2 - x1)>10) and (abs(y2 - y1)>10):

    j = 0
    n = 0
    bit = 2
    while n <= i:
        z = 0
        while j != 0 and z < j:
            if abs(xs1[j] - xs1[z]) < 20 and abs(xs2[j] - xs2[z]) < 20 and abs(ys1[j] - ys1[z]) < 20 and abs(
                    ys2[j] - ys2[z]) < 20:  # 剔除过近的端点值
                del xs1[j]
                del xs2[j]
                del ys1[j]
                del ys2[j]
                bit = 1
                break
            else:
                bit = 0
            z = z + 1
        if j == 0:
            j = j + 1
        if n != 0 and bit == 0:
            j = j + 1
        if bit == 1:
            i = i - 1

        n = n + 1

    j = 0
    while j < i:
        cv2.line(img, (xs1[j], ys1[j]), (xs2[j], ys2[j]), (0, 0, 255), 2)
        dig[j] = (math.atan2(ys2[j] - ys1[j], xs2[j] - xs1[j]) * 180) / 3.1415
        j = j + 1

    return dig


i = 0
photoname = 1

book = xlwt.Workbook(encoding="ascii", style_compression=0)
sh = book.add_sheet(sheetname="sheet_1", cell_overwrite_ok=False)
sh.row(0).height = 20 * 30  # 直接设置第一行行高为30磅，20为基数
# 通过wlwt.XFStyle()对象设置复杂样式
style = xlwt.XFStyle()  # 获取样式对象
font = xlwt.Font()  # 获取字体对象
font.name = '微软雅黑'  # 设置字体类型
style.font = font  # 将设置后的字体对象赋值给样式对象的font属性
sh.write(r=0, c=0, label='时间', style=style)  # sytle为第3步设置的样式对象
sh.write(r=0, c=1, label='角度', style=style)  # sytle为第3步设置的样式对象
i = 0
photoname = 1

while (1):
    i = i + 1
    # get a frame
    ret, frame = cap.read()
    # show a fram
    cv2.imshow('Image', frame)  # 生成摄像头窗口

    if i == 20:  # 定时装置，定时截屏，可以修改。
        filename = str(photoname) + '.png'  # filename为图像名字，将photoname作为编号命名保存的截图
        now = time.localtime()
        now_time = time.strftime("%Y-%m-%d %H:%M:%S", now)
        cv2.imwrite(r'D:\test' + '\\' + filename, frame)  # 截图 前面为放在桌面的路径 frame为此时的图像
        dig = recognition(r'D:\test' + '\\' + filename)
        angel = (dig[0] + dig[1]) / 2
        cv2.putText(frame, "{:.2f}".format(angel), (5, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
        cv2.imshow('Image', frame)  # 生成摄像头窗口
        cv2.waitKey(50)
        print(filename + '保存成功')  # 打印保存成功
        print(now_time)
        print(angel)  # 输出角度值
        i = 0  # 清零
        sh.write(r=photoname, c=0, label=now_time, style=style)  # sytle为第3步设置的样式对象
        sh.write(r=photoname, c=1, label=angel, style=style)  # sytle为第3步设置的样式对象
        photoname = photoname + 1

        if photoname >= 20:  # 最多截图20张 然后退出（如果调用photoname = 1 不用break为不断覆盖图片）
            break

    if cv2.waitKey(1) & 0xff == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
book.save("test.xls")