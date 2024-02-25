import cv2
import numpy as np


def get_red(img, rect):
    # 计算矩形四个顶点坐标
    center, size, angle = rect
    rect = (center, size, angle)
    box = cv2.boxPoints(rect)
    box = np.int0(box)

    # 计算矩形区域红色像素的比例
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [box], 0, 255, -1)
    red_mask = cv2.inRange(img, (0, 0, 150), (80, 80, 255))
    red_pixels = cv2.countNonZero(red_mask & mask)
    total_pixels = cv2.countNonZero(mask)
    red_ratio = red_pixels / total_pixels

    return red_ratio
'''输入一个视频
    输出识别好的角度及处理过的视频'''
def my_function(video_file):
    # 打开视频文件
    cap = cv2.VideoCapture(video_file)

    # 读取视频的第一帧
    ret, frame = cap.read()

    # 获取视频帧的宽度和高度
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 创建VideoWriter对象，用于将识别出的角度写入视频
    out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(*'MJPG'), 30, (width, height))


    num=0
    while True:
        num+=1
        # 读取视频的下一帧
        ret, frame = cap.read()

        # 高斯滤波处理腐蚀再膨胀后的图像
        dilation_gauss = cv2.GaussianBlur(frame, (7, 7), 0, 0)
        cv2.imshow("chuli",frame)
        # 如果无法读取下一帧，则退出循环
        if not ret:
            break

        # 将视频帧转换为灰度图像
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 进行Canny边缘检测
        edges = cv2.Canny(gray, 100, 200)

        # 查找轮廓
        contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 对于每个轮廓
        for contour in contours:
            # 计算轮廓的最小外接矩形
            rect = cv2.minAreaRect(contour)

            # 提取矩形的中心坐标、宽度、高度和旋转角度
            center, size, angle = rect
            width_rect,high_rect = size
            # 筛选识别条
            # 设置颜色占比阈值
            red=get_red(frame,rect)
            angle=90-angle
            # 设置面积阈值
            spuare_rect= width_rect*high_rect

            # 将旋转角度打印出来
            if red>=0.6 and spuare_rect>6500:
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                cv2.drawContours(frame, [box], 0, (0, 0, 255), 2)
                cv2.putText(frame, "Object angle: {:.2f}".format(angle), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (0, 0, 255), 2)
                if num%5==0:
                    print("Object angle: {:.1f}".format(angle))

        # 将处理后的帧写入输出视频
        out.write(frame)
        cv2.imshow("frame", frame)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()