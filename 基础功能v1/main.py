import sys
import cv2 as cv
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from ui import Ui_mainWindow
from PyQt5.QtCore import Qt
import func2
import datetime
import matplotlib.pyplot as plt

class EmittingStr(QtCore.QObject):
    textWritten = QtCore.pyqtSignal(str)  # 定义一个发送str的信号

    def write(self, text):
        self.textWritten.emit(str(text))


class mainWindow(QtWidgets.QMainWindow, Ui_mainWindow):
    def __init__(self, parent=None):
        super(mainWindow, self).__init__(parent)
        self.timer_video = QtCore.QTimer()
        self.setupUi(self)
        self.init_logo()
        self.init_slots()
        self.cap = cv.VideoCapture()
        self.out = None
        sys.stdout = EmittingStr(textWritten=self.outputWritten)
        sys.stderr = EmittingStr(textWritten=self.outputWritten)

    def init_slots(self):
        self.fileButton.clicked.connect(self.button_image_open)
        self.rtspButton.clicked.connect(self.button_video_open)
        self.cameraButton.clicked.connect(self.button_camera_open)
        self.timer_video.timeout.connect(self.show_video_frame)

    def init_logo(self):
        pix = QtGui.QPixmap('')  # 绘制初始化图片
        self.label.setScaledContents(True)
        self.label.setPixmap(pix)

    def button_image_open(self):
        print('打开图片')
        name_list = []

        img_name, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "打开图片", "", "*.jpg;;*.png;;All Files(*)")
        print(img_name)
        if not img_name:
            return

        img = cv.imread(img_name)
        kernel = np.ones((5, 5), np.uint8)
        erosion = cv.erode(img, kernel, iterations=5)
        dilation = cv.dilate(erosion, kernel, iterations=5)
        # 高斯滤波处理腐蚀再膨胀后的图像
        dilation_gauss = cv.GaussianBlur(dilation, (7, 7), 0, 0)
        # 用cv自带的Canny函数处理，后面是阈值上界和下界

        edge_img = edge_img = cv.Canny(dilation_gauss, 62, 195)

        mask = np.zeros_like(edge_img)  # 获取一个与edge_img相同的数组
        mask = cv.fillPoly(mask, np.array([[[135, 356], [135, 211], [249, 211], [249, 356]]]),
                           color=255)  # 提取mask里感兴趣的掩码，
        # 参数1数组，参数2感兴趣区域对应点，参数3掩码灰度值

        masked_edge_img = cv.bitwise_and(edge_img, mask)  # 布尔运算，参数1原图，参数2掩码

        edge_img1 = masked_edge_img

        lines = cv.HoughLinesP(edge_img1, 1, np.pi / 180, 15, minLineLength=40, maxLineGap=20)

        # print(lines)
        a = lines[0]
        b = lines[1]
        c = lines[2]
        a1 = a.ravel()
        b1 = b.ravel()
        c1 = c.ravel()
        a2 = tuple(a1)
        b2 = tuple(b1)
        c2 = tuple(c1)

        cv.line(img, (a2[0], a2[1]), (a2[2], a2[3]), color=(0, 255, 255), thickness=3)
        cv.line(img, (b2[0], b2[1]), (b2[2], b2[3]), color=(0, 255, 255), thickness=3)

        # print(img)

        showimg = img

        self.result = cv.cvtColor(showimg, cv.COLOR_BGR2BGRA)
        self.result = cv.resize(self.result, (640, 480), interpolation=cv.INTER_AREA)
        self.QtImg = QtGui.QImage(self.result.data, self.result.shape[1], self.result.shape[0],
                                  QtGui.QImage.Format_RGB32)
        self.labelout.setPixmap(QtGui.QPixmap.fromImage(self.QtImg))
        self.labelout.setScaledContents(True)

    def button_video_open(self):
        print('打开视频')
        video_name, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "打开视频", "", "*.mp4;;*.avi;;All Files(*)")
        print(video_name)

        if not video_name:
            return

        cap = cv.VideoCapture(video_name)

        # 读取视频的第一帧
        ret, frame = cap.read()

        # 获取视频帧的宽度和高度
        width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

        # 创建VideoWriter对象，用于将识别出的角度写入视频
        out = cv.VideoWriter('/result/output.avi', cv.VideoWriter_fourcc(*'MJPG'), 30, (width, height))

        num = 0

        angle = 0
        # plt.ion()  # 开启interactive mode 成功的关键函数
        # plt.figure(1)
        # t = [0]
        # t_now = 0
        # angle_list = [0]

        while True:
            num += 1
            # 读取视频的下一帧
            ret, frame = cap.read()

            # 高斯滤波处理
            dilation_gauss = cv.GaussianBlur(frame, (7, 7), 0, 0)
            # cv.imshow("chuli", frame)
            # 如果无法读取下一帧，则退出循环
            #if not ret:
                #break

            # 将视频帧转换为灰度图像
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

            # 进行Canny边缘检测
            edges = cv.Canny(gray, 100, 200)

            # 查找轮廓
            contours, hierarchy = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)


            # 对于每个轮廓
            for contour in contours:
                # 计算轮廓的最小外接矩形
                rect = cv.minAreaRect(contour)

                # 提取矩形的中心坐标、宽度、高度和旋转角度
                center, size, angle = rect
                width_rect, high_rect = size
                # 筛选识别条
                # 设置颜色占比阈值
                red = func2.get_red(frame, rect)
                angle = 88.6 - angle
                # 设置面积阈值
                spuare_rect = width_rect * high_rect

                # 将旋转角度打印出来
                if red >= 0.6 :
                    box = cv.boxPoints(rect)
                    box = np.int0(box)
                    cv.drawContours(frame, [box], 0, (0, 0, 255), 2)
                    cv.putText(frame, "Object angle: {:.2f}".format(angle), (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1,
                               (0, 0, 255), 2)
                    if num % 5 == 0:
                        if angle < 45 and angle >0:
                            datetime_object = datetime.datetime.now()
                            print("当前时间：{}".format(datetime_object))
                            print("角度：{:.1f}".format(angle))
                            print("-----------")

                        # plt.clf()
                        # t_now = angle
                        # t.append(num)  # 模拟数据增量流入，保存历史数据
                        # angle_list.append(t_now)  # 模拟数据增量流入，保存历史数据
                        # plt.plot(t, angle_list, '-r')
                        # plt.pause(0.01)





            self.result = cv.cvtColor(frame, cv.COLOR_BGR2BGRA)
            self.result = cv.resize(self.result, (368, 640), interpolation=cv.INTER_AREA)
            self.QtImg = QtGui.QImage(self.result.data, self.result.shape[1], self.result.shape[0],
                                      QtGui.QImage.Format_RGB32)
            self.labelout.setPixmap(QtGui.QPixmap.fromImage(self.QtImg))
            #self.labelout.setScaledContents(True)
            self.labelout.setAlignment(Qt.AlignCenter)
            # 将处理后的帧写入输出视频
            out.write(frame)
            # cv.imshow("frame", frame)

            c = cv.waitKey(100)

            if c == 27:  # 27是ESC键，按ESC退出
                break

        cap.release()
        cv.destroyAllWindows()

        # flag = self.cap.open(frame)
        # print(flag)
        # if flag == False:
        #     QtWidgets.QMessageBox.warning(
        #         self, u"Warning", u"打开视频失败", buttons=QtWidgets.QMessageBox.Ok, defaultButton=QtWidgets.QMessageBox.Ok)
        # else:
        #     self.out = cv.VideoWriter('result/vedio_prediction.avi', cv.VideoWriter_fourcc(
        #         *'MJPG'), 20, (int(self.cap.get(3)), int(self.cap.get(4))))
        #     self.timer_video.start(30)
        #     # self.rtspButton.setDisabled(True)
        #     self.fileButton.setDisabled(True)
        #     self.cameraButton.setDisabled(True)

    def button_camera_open(self):
        if not self.timer_video.isActive():
            # 默认使用第一个本地camera
            flag = self.cap.open(0)
            if flag == False:
                QtWidgets.QMessageBox.warning(
                    self, u"Warning", u"打开摄像头失败", buttons=QtWidgets.QMessageBox.Ok,
                    defaultButton=QtWidgets.QMessageBox.Ok)
            else:
                self.out = cv.VideoWriter('result/camera_prediction.avi', cv.VideoWriter_fourcc(
                    *'MJPG'), 20, (int(self.cap.get(3)), int(self.cap.get(4))))
                self.timer_video.start(30)
                self.rtspButton.setDisabled(True)
                self.fileButton.setDisabled(True)
                self.cameraButton.setText(u"关")
        else:
            self.timer_video.stop()
            self.cap.release()
            self.out.release()
            self.label.clear()
            self.init_logo()
            self.rtspButton.setDisabled(False)
            self.fileButton.setDisabled(False)
            self.cameraButton.setText(u"开")

    def show_video_frame(self):
        # name_list = []

        flag, img = self.cap.read()
        if img is not None:
            showimg = img

            self.out.write(showimg)
            show = cv.resize(showimg, (640, 480))
            self.result = cv.cvtColor(show, cv.COLOR_BGR2RGB)
            showImage = QtGui.QImage(self.result.data, self.result.shape[1], self.result.shape[0],
                                     QtGui.QImage.Format_RGB888)
            self.labelout.setPixmap(QtGui.QPixmap.fromImage(showImage))
            self.labelout.setScaledContents(True)
            # print('123455475475675')

        else:
            self.timer_video.stop()
            self.cap.release()
            self.out.release()
            self.labelout.clear()
            self.rtspButton.setDisabled(False)
            self.fileButton.setDisabled(False)
            self.cameraButton.setDisabled(False)
            self.init_logo()

    def outputWritten(self, text):
        cursor = self.textBrowser.textCursor()
        cursor.movePosition(QtGui.QTextCursor.End)
        cursor.insertText(text)
        self.textBrowser.setTextCursor(cursor)
        self.textBrowser.ensureCursorVisible()

    def retranslateUi(self, mainWindow):
        _translate = QtCore.QCoreApplication.translate
        mainWindow.setWindowTitle(_translate("mainWindow", "一种基于机器视觉的机械臂旋转角度识别系统"))
        self.label_4.setText(_translate("mainWindow", "一种基于机器视觉的机械臂旋转角度识别系统"))
        self.label_5.setText(_translate("mainWindow", "<html><head/><body><p>设置</p></body></html>"))
        self.label_10.setText(_translate("mainWindow", "input"))
        self.fileButton.setToolTip(_translate("mainWindow", "file"))
        self.cameraButton.setToolTip(_translate("mainWindow", "camera"))
        self.rtspButton.setToolTip(_translate("mainWindow", "rtsp"))
        self.label_11.setText(_translate("mainWindow", "结果"))
        self.label_6.setText(_translate("mainWindow", "视频 video"))


import picture_rc

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    myWin = mainWindow()
    myWin.show()
    sys.exit(app.exec_())
