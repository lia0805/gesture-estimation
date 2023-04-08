import cv2
import mediapipe as mp
import numpy as np


def calculate_angle(a, b, c):
    '''
    计算角度
    :param a:
    :param b:
    :param c:
    :return:
    '''

    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle


def calculate_dist(a, b):
    '''
    计算欧式距离
    :param a:
    :param b:
    :return:
    '''

    a = np.array(a)
    b = np.array(b)
    dist = np.linalg.norm(a - b)
    return dist


if __name__ == '__main__':

    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    cap = cv2.VideoCapture('videos/1.mp4')

    # 分辨率
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    # 保存结果视频
    out = cv2.VideoWriter("result.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 30, (frame_width, frame_height))

    counter = 0
    stage = None

    with mp_pose.Pose(min_detection_confidence=0.3, min_tracking_confidence=0.8) as pose:
        while cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                break

            # 转换下颜色空间
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # 这里设置为不可写
            image.flags.writeable = False

            # 检测
            results = pose.process(image)

            # 这里设置为可写，颜色也转换回去
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # 提取关键点
            try:
                landmarks = results.pose_landmarks.landmark

                # 获取相应关键点的坐标
                lshoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                             landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                lelbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                          landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                lwrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                          landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                lhip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                lankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                          landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                lknee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]

                rshoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                             landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                relbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                          landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                rwrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                          landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
                rhip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                        landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                rankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                          landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
                rknee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                         landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]

                # 计算角度
                langle = calculate_angle(lshoulder, lelbow, lwrist)
                rangle = calculate_angle(rshoulder, relbow, rwrist)
                lsangle = calculate_angle(lhip, lshoulder, lelbow)
                rsangle = calculate_angle(rhip, rshoulder, relbow)
                ankdist = calculate_dist(lankle, rankle)
                rwdist = calculate_dist(rhip, rwrist)
                lwdist = calculate_dist(lhip, lwrist)
                rhangle = calculate_angle(rshoulder, rhip, rknee)
                lhangle = calculate_angle(lshoulder, lhip, lknee)
                rkangle = calculate_angle(rankle, rknee, rhip)
                lkangle = calculate_angle(lankle, lknee, lhip)
                # 这块是具体的业务逻辑，各个数值，可根据自己实际情况适当调整
                if ((rhangle > 80 and lhangle > 80) and (rhangle < 110 and lhangle < 110) and (
                        lkangle < 100 and rkangle < 100)):
                    stage = 'sitting'

                elif (langle < 160 and langle > 40) or (rangle < 160 and rangle > 40):
                    if ((lsangle > 20 or rsangle > 20) and (lwdist > 0.3 or rwdist > 0.3)):
                        stage = "wave"

                elif ((ankdist > 0.084) and (langle > 150) and (rangle > 150)):
                    counter += 1
                    if counter > 1:
                        stage = 'walking'
                else:
                    stage = 'standing'
                    counter = 0
            except:
                pass

            cv2.rectangle(image, (0, 0), (225, 73), (245, 117, 16), -1)
            cv2.putText(image, 'STAGE', (65, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, stage, (60, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

            # 画骨骼关键点
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                      mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                                      )

            # 显示结果帧
            cv2.imshow('mediapipe demo', image)

            # 保存结果帧
            out.write(image)

            # 按q退出
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        # 资源释放
        cap.release()
        cv2.destroyAllWindows()