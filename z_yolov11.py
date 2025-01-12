import copy

from ultralytics import YOLO
import cv2
import threading
import time
import os
import socket

from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np


def z_udp(send_data, address):
    # 1. 创建udp套接字
    udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    # 2. 准备接收方的地址
    # dest_addr = ('127.0.0.1', 8080)
    # 4. 发送数据到指定的电脑上
    udp_socket.sendto(send_data.encode('utf-8'), address)
    # 5. 关闭套接字
    udp_socket.close()


# 上面是http处理
def load_area():  # 初始化区域
    global area_Code
    for key in area_Code.keys():
        track_file = f"./{key}.txt"
        if os.path.exists(track_file):  # 存在就加载数据对应赛道数据
            with open(track_file, 'r') as file:
                content = file.read()
            lines = content.split('\n')
            for line in lines:
                if line:
                    polgon_array = {'coordinates': [], 'code': 0, 'direction': 0}
                    paths = line.split(' ')
                    if len(paths) < 2:
                        print("分区文件错误！")
                        return
                    items = paths[0].split(',')
                    for item in items:
                        if item:
                            x, y = item.split('/')
                            polgon_array['coordinates'].append((int(x), int(y)))
                    polgon_array['code'] = int(paths[1])
                    if len(paths) > 2:
                        polgon_array['direction'] = int(paths[2])
                    area_Code[key].append(polgon_array)


def deal_area(ball_array, img, code):  # 处理该摄像头内区域
    ball_area_array = []
    for ball in ball_array:
        x = (ball[0] + ball[2]) / 2
        y = (ball[1] + ball[3]) / 2
        point = (x, y)
        if code in area_Code.keys():
            for area in area_Code[code]:
                pts = np.array(area['coordinates'], np.int32)
                Result = cv2.pointPolygonTest(pts, point, False)  # -1=在外部,0=在线上，1=在内部
                if Result > -1.0:
                    ball.append(area['code'])
                    ball.append(area['direction'])
                    ball_area_array.append(ball)
    if len(ball_area_array) != 0:
        area_array = []
        for ball in ball_area_array:
            if ball[6] not in area_array:  # 记录所有被触发的多边形号码
                area_array.append(ball[6])
        for area in area_Code[code]:  # 遍历该摄像头所有区域
            pts = np.array(area['coordinates'], np.int32)
            if area['code'] in area_array:
                polygonColor = (255, 0, 255)
            else:
                polygonColor = (0, 255, 255)
            cv2.polylines(img, [pts], isClosed=True, color=polygonColor, thickness=8)
    return ball_area_array, img


def deal_simple():
    global camera_frame_array
    global run_flg
    color = (0, 255, 0)
    model = YOLO("best.pt")
    # model = myTr.Detector(model_path=b"./best8.engine", dll_path="./trt/yolov8.dll")
    names = {0: 'yellow', 1: 'blue', 2: 'red', 3: 'purple', 4: 'orange', 5: 'green', 6: 'Brown',
             7: 'black',
             8: 'pink', 9: 'White', 10: 'xx_s_yello', 11: 'xx_s_white', 12: 'xx_s_red',
             13: 'xx_s_black'}

    while True:
        if not run_flg:  # 倒计时运行标志
            continue
        if time.time() > run_time:
            run_flg = False
        integration_qiu_array = []
        for cap_num in range(0, len(cap_array)):
            ret, frame = cap_array[cap_num].read()
            if not ret:
                print("读取帧失败")
                continue

            # result = model.predict(frame)
            # results = model.visualize(result)
            results = model.predict(source=frame, show=False, conf=0.5, iou=0.45, imgsz=1280)
            qiu_array = []
            if len(results) != 0:  # 整合球的数据
                # names = results[0].names
                result = results[0].boxes.data

                for r in result:
                    if int(r[5].item()) < 10:
                        array = [int(r[0].item()), int(r[1].item()), int(r[2].item()), int(r[3].item()),
                                 round(r[4].item(), 2), names[int(r[5].item())]]
                        cv2.rectangle(frame, (array[0], array[1]), (array[2], array[3]), color, thickness=3)
                        cv2.putText(frame, "%s %s" % (array[5], str(array[4])), (array[0], array[1] - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    fontScale=1,
                                    color=(0, 0, 255), thickness=2)
                        qiu_array.append(array)
            if len(qiu_array):  # 处理范围内跟排名
                # print("处理范围内排名")
                qiu_array, frame = deal_area(qiu_array, frame, cap_num)  # 统计各个范围内的球，并绘制多边形
                camera_frame_array[cap_num] = frame
            if len(qiu_array) > 0:
                integration_qiu_array.extend(qiu_array)
                z_udp(str(integration_qiu_array), server_self_rank)  # 发送数据s
            else:
                camera_frame_array[cap_num] = frame
        # if len(integration_qiu_array) > 0:
        #     integration_qiu_array = filter_max_value(integration_qiu_array)
        #     z_udp(str(integration_qiu_array), server_self_rank)  # 发送数据s


def show_map():
    global run_flg
    target_width, target_height = 960, 540
    show_flg = False  # 图像识别显示标志

    while True:
        if not run_flg:
            cv2.destroyAllWindows()
            show_flg = False
            # time.sleep(5)
            # run_flg = True
            continue
        if not show_flg:
            if cv2.getWindowProperty('display', cv2.WND_PROP_VISIBLE) < 1:
                cv2.namedWindow("display", cv2.WINDOW_NORMAL)
                cv2.resizeWindow("display", 1100, 1200)
                show_flg = True
        canvas = np.zeros((1080 + target_height * 2, 1920, 3), dtype=np.uint8)  # 三元色，对应的三维数组
        canvas[0:target_height, 0: target_width] = cv2.resize(camera_frame_array[0],
                                                              (target_width, target_height))
        canvas[target_height:1080, 0: target_width] = cv2.resize(camera_frame_array[1],
                                                                 (target_width, target_height))
        canvas[1080:1080 + target_height, 0: target_width] = cv2.resize(camera_frame_array[2],
                                                                        (target_width, target_height))
        canvas[1080 + target_height:2160, 0: target_width] = cv2.resize(camera_frame_array[6],
                                                                        (target_width, target_height))
        canvas[0:target_height, target_width: 1920] = cv2.resize(camera_frame_array[3],
                                                                 (target_width, target_height))
        canvas[target_height:1080, target_width: 1920] = cv2.resize(camera_frame_array[4],
                                                                    (target_width, target_height))
        canvas[1080:1080 + target_height, target_width: 1920] = cv2.resize(camera_frame_array[5],
                                                                           (target_width, target_height))
        canvas[1080 + target_height:2160, target_width: 1920] = cv2.resize(camera_frame_array[7],
                                                                           (target_width, target_height))

        cv2.imshow("display", canvas)

        key = cv2.waitKey(1)
        if key == 27:  # 如果按下ESC键，退出循环
            run_flg = not run_flg

    cv2.destroyAllWindows()


class SimpleHTTPRequestHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/html; charset=utf-8')
        self.end_headers()
        self.wfile.write('你对HTTP服务端发送了POST'.encode('utf-8'))
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length).decode('utf-8')
        print("客户端发送的post内容=" + post_data)
        if post_data == "start":
            self.handle_start_command()
        if post_data == "stop":
            self.handle_stop_command()

    def handle_start_command(self):
        global run_flg
        global run_time
        run_flg = True
        run_time = time.time() + 600
        print('执行开始')

    def handle_stop_command(self):
        print('执行停止')


if __name__ == "__main__":
    server_self_rank = ("192.168.0.59", 8080)
    camera_num = 8
    area_Code = {}  # 摄像头代码列表
    load_area()  # 初始化区域划分

    run_flg = True  # 图像识别运行标志
    run_time = time.time() + 600  # 空运行时间，超过这个时间没收到客户机信号，即停止推理

    cap_array = {}  # 摄像头数组
    camera_frame_array = {}  # 摄像头图片数组

    for i in range(0, camera_num):
        cap_array[i] = None
        camera_frame_array[i] = None

    # 显示线程
    show_thread = threading.Thread(target=show_map, daemon=True)
    show_thread.start()

    # 启动 HTTPServer 接收外部命令控制本程序
    httpServer_addr = ('0.0.0.0', 8081)  # 接收网络数据包控制
    httpd = HTTPServer(httpServer_addr, SimpleHTTPRequestHandler)
    http_thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    http_thread.start()

    # 推理主线程
    run_thread = threading.Thread(target=deal_simple)
    run_thread.start()

    # run_thread = {}  # 多线程运行，推理效率暴降
    # for i in range(0, camera_num):
    #     run_thread[i] = threading.Thread(target=deal_threads, args=(cap_array[i], i))
    #     run_thread[i].start()
