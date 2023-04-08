#导入库
import cv2
import mediapipe as mp
from tqdm import tqdm
import time
#导入模型
mp_pose=mp.solutions.pose
mp_drawing=mp.solutions.drawing_utils
pose=mp_pose.Pose(static_image_mode=False,#选择静态图片还是连续视频帧
                 model_complexity=2,#选择人体姿态关键点检测模型，0性能差但快，2性能好但慢，1介于之间
                 smooth_landmarks=True,#是否选择平滑关键点
                 min_detection_confidence=0.5,#置信度阈值
                 min_tracking_confidence=0.5)#追踪阈值

def process_frame(img):
    #BGR转RGB
    img_RGB=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    #RGB图像输入模型，获取预测结果
    result=pose.process(img_RGB)
    #可视化
    mp_drawing.draw_landmarks(img,result.pose_landmarks,mp_pose.POSE_CONNECTIONS)
    return img

import cv2
import time
#获取摄像头，0表示系统默认摄像头
cap=cv2.VideoCapture(0)
#打开cap
cap.open(0)
#无限循环，直到break
while cap.isOpened():
    #获取画面
    success,frame=cap.read()
    if not success:
        print('Error')
        break
    #处理帧函数
    frame=process_frame(frame)
    #展示处理后的三通道图像
    cv2.imshow('my_window',frame)
    if cv2.waitKey(1) in [ord('q'),27]: #q或esc退出
        break
#关闭摄像头
cap.release()
#关闭图像窗口
cv2.destroyAllWindows()

