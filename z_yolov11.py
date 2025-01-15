from typing import Any

import win32gui
import win32ui
import win32con

from ultralytics import YOLO
import cv2
import threading
import time
import os
import socket

from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np


def enum_windows_callback(hwnd, windows_list):
    """
    枚举顶层窗口的回调函数，用于收集带有₮字符的窗口句柄和标题。
    """
    title = win32gui.GetWindowText(hwnd)
    if "₮" in title:  # 筛选包含 ₮ 字符的窗口
        windows_list.append((hwnd, title))


def get_windows_with_special_char():
    """
    获取所有包含 ₮ 字符的窗口句柄和标题。
    """
    windows = []
    win32gui.EnumWindows(enum_windows_callback, windows)
    return windows


def capture_window_as_opencv(hwnd):
    """
    截取指定窗口并直接返回 OpenCV 图像。
    :param hwnd: 窗口句柄
    :return: OpenCV 图像 (BGR 格式)
    """
    left, top, right, bottom = win32gui.GetWindowRect(hwnd)
    width = right - left
    height = bottom - top

    hwnd_dc = win32gui.GetWindowDC(hwnd)
    mfc_dc = win32ui.CreateDCFromHandle(hwnd_dc)
    save_dc = mfc_dc.CreateCompatibleDC()
    save_bitmap = win32ui.CreateBitmap()
    save_bitmap.CreateCompatibleBitmap(mfc_dc, width, height)
    save_dc.SelectObject(save_bitmap)

    save_dc.BitBlt((0, 0), (width, height), mfc_dc, (0, 0), win32con.SRCCOPY)
    bmp_bits = save_bitmap.GetBitmapBits(True)

    img_array = np.frombuffer(bmp_bits, dtype=np.uint8)
    img_array = img_array.reshape((height, width, 4))  # BGRA 格式
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_BGRA2BGR)

    win32gui.DeleteObject(save_bitmap.GetHandle())
    save_dc.DeleteDC()
    mfc_dc.DeleteDC()
    win32gui.ReleaseDC(hwnd, hwnd_dc)

    return img_bgr


def images_are_different(image1, image2, threshold=5000):
    """
    比较两张图像是否不同。
    :param image1: 第一张图像
    :param image2: 第二张图像
    :param threshold: 差异值阈值
    :return: 是否不同 (布尔值)
    """
    if image1 is None or image2 is None:
        return True  # 如果其中一张图像为空，认为不同

    diff = cv2.absdiff(image1, image2)  # 计算差异图像
    diff_sum = np.sum(diff)  # 计算差异值的总和
    return diff_sum > threshold  # 判断是否超过阈值


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
    # 用于存储每个窗口的前一张图片
    previous_images = {}
    while True:
        if not run_flg:  # 倒计时运行标志
            time.sleep(0.1)  # 等待 100 毫秒
            continue

        # 获取所有包含 ₮ 的窗口
        special_windows = get_windows_with_special_char()
        if not special_windows:
            print("未找到包含 ₮ 字符的窗口。")
            continue

        for idx, (hwnd, title) in enumerate(special_windows):
            try:
                # 截取当前窗口的图片
                current_image = capture_window_as_opencv(hwnd)

                # 获取前一张图片
                prev_image = previous_images.get(hwnd)

                # 如果没有前一张图片，或者图片不同，则保存
                if prev_image is None or images_are_different(prev_image, current_image):
                    results = model.predict(source=current_image, show=False, conf=0.3, iou=0.2, imgsz=1472)
                    qiu_array = []
                    if len(results) != 0:  # 整合球的数据
                        names = results[0].names
                        result = results[0].boxes.data

                        for r in result:

                            array = [int(r[0].item()), int(r[1].item()), int(r[2].item()), int(r[3].item()),
                                     round(r[4].item(), 2), names[int(r[5].item())]]
                            cv2.rectangle(current_image, (array[0], array[1]), (array[2], array[3]), color,
                                          thickness=1)
                            cv2.putText(current_image, "%s" % (array[5]),
                                        (array[0], array[1] - 5),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        fontScale=0.5,
                                        color=(0, 0, 255), thickness=1)
                            qiu_array.append(array)



                    cv2.imshow("YOLO Inference Result", current_image)
                    camera_frame_array[idx] = current_image
                    print(qiu_array)

                    # 更新前一张图片
                    previous_images[hwnd] = current_image
                else:
                    print(f"窗口 {title} 的内容未变化，跳过保存。")
            except Exception as e:
                print(f"无法截取窗口 {title}，错误: {e}")
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cv2.destroyAllWindows()
    # result = model.predict(frame)
    # results = model.visualize(result)

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
        canvas = np.zeros((1080 + target_height, 1920, 3), dtype=np.uint8)  # 三元色，对应的三维数组
        if not (camera_frame_array[0] is None):
            canvas[0:928, 0: 1280] = cv2.resize(camera_frame_array[0], (1280, 928))
        # canvas[target_height:1080, 0: target_width] = cv2.resize(camera_frame_array[1],
        #                                                          (target_width, target_height))
        # canvas[1080:1080 + target_height, 0: target_width] = cv2.resize(camera_frame_array[2],
        #                                                                 (target_width, target_height))
        # canvas[1080 + target_height:2160, 0: target_width] = cv2.resize(camera_frame_array[6],
        #                                                                 (target_width, target_height))
        # canvas[0:target_height, target_width: 1920] = cv2.resize(camera_frame_array[3],
        #                                                          (target_width, target_height))
        # canvas[target_height:1080, target_width: 1920] = cv2.resize(camera_frame_array[4],
        #                                                             (target_width, target_height))
        # canvas[1080:1080 + target_height, target_width: 1920] = cv2.resize(camera_frame_array[5],
        #                                                                    (target_width, target_height))
        # canvas[1080 + target_height:2160, target_width: 1920] = cv2.resize(camera_frame_array[7],
        #                                                                    (target_width, target_height))

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
        camera_frame_array[i] = np.zeros((100, 100, 3), dtype=np.uint8)

    # 显示线程
    # show_thread = threading.Thread(target=show_map, daemon=True)
    # show_thread.start()

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
