import mediapipe as mp
import cv2
import time # 为了计算fps（每秒传输帧数）的值


pTime = 0 # 初始化一个变量
cap = cv2.VideoCapture('videos/1.mp4') # 调用摄像头

#读取成功后，初始化一些要用的人体姿态关键点的框架
mpPose = mp.solutions.pose
pose = mpPose.Pose()
# 接下来初始化一些绘图工具
mpDraw = mp.solutions.drawing_utils


while True:
    success, img = cap.read()# 读取视频每一帧画面
    # 开始处理视频 先将图片转化成rgb格式
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(imgRGB)# 处理一下这个图片
    # 如果检测到这个关键点的信息,就将它刻画出来
    if results.pose_landmarks:
        mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)# 打印出人体的关键点，再将它们连起来

    # 对某一个关键点进行操作，遍历一下33个关键点
        for id, lm in enumerate(results.pose_landmarks.landmark):
            h, w, c = img.shape #获取视频的高，宽，通道数
            cx, cy = int(lm.x * w), int(lm.y * h)#计算c坐标和y坐标
            print(id, cx, cy)

    # 先刻画一下fps的值
    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime
    # 将fps的值刻画到视频中,并将fps的值转化为int类型,写下坐标，字体样式，字体大小，颜色，字体厚度
    cv2.putText(img, str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN, 3, (255,255,0), 3)

    cv2.imshow("image", img)
    cv2.waitKey(1)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break