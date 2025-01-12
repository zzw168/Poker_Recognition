import win32gui
import win32ui
import win32con
import win32api
from PIL import Image, ImageChops
import time
import os

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

def capture_window_as_image(hwnd):
    """
    截取指定窗口并返回 PIL 图像对象。
    :param hwnd: 窗口句柄
    :return: 截图的 PIL Image 对象
    """
    # 获取窗口设备上下文 (DC)
    left, top, right, bottom = win32gui.GetWindowRect(hwnd)
    width = right - left
    height = bottom - top

    hwnd_dc = win32gui.GetWindowDC(hwnd)
    mfc_dc = win32ui.CreateDCFromHandle(hwnd_dc)
    save_dc = mfc_dc.CreateCompatibleDC()
    save_bitmap = win32ui.CreateBitmap()
    save_bitmap.CreateCompatibleBitmap(mfc_dc, width, height)
    save_dc.SelectObject(save_bitmap)

    # 截图并保存为临时 BMP 文件
    save_dc.BitBlt((0, 0), (width, height), mfc_dc, (0, 0), win32con.SRCCOPY)
    bmp_info = save_bitmap.GetInfo()
    bmp_data = save_bitmap.GetBitmapBits(True)

    # 将位图数据转换为 PIL 图像对象
    img = Image.frombuffer(
        "RGB",
        (bmp_info["bmWidth"], bmp_info["bmHeight"]),
        bmp_data,
        "raw",
        "BGRX",
        0,
        1,
    )

    # 释放资源
    win32gui.DeleteObject(save_bitmap.GetHandle())
    save_dc.DeleteDC()
    mfc_dc.DeleteDC()
    win32gui.ReleaseDC(hwnd, hwnd_dc)

    return img

def images_are_different(img1, img2):
    """
    比较两张图片是否不同。
    :param img1: 第一张 PIL 图像
    :param img2: 第二张 PIL 图像
    :return: 如果不同，返回 True；否则返回 False
    """
    time_old = time.time()
    diff = ImageChops.difference(img1, img2)
    time_all = time.time() - time_old
    print(time_all)
    print(time_old)
    return diff.getbbox() is not None

if __name__ == "__main__":
    # 获取所有包含 ₮ 的窗口
    special_windows = get_windows_with_special_char()

    if not special_windows:
        print("未找到包含 ₮ 字符的窗口。")
    else:
        print(f"找到 {len(special_windows)} 个包含 ₮ 字符的窗口。")

        # 用于存储每个窗口的前一张图片
        previous_images = {}

        try:
            while True:  # 无限循环
                for idx, (hwnd, title) in enumerate(special_windows):
                    print(f"正在截图窗口: {title}")
                    try:
                        # 截取当前窗口的图片
                        current_image = capture_window_as_image(hwnd)

                        # 获取前一张图片
                        prev_image = previous_images.get(hwnd)

                        # 如果没有前一张图片，或者图片不同，则保存
                        if prev_image is None or images_are_different(prev_image, current_image):
                            # 保存图片
                            timestamp = time.strftime("%Y%m%d_%H%M%S")  # 格式：20250110_121030
                            save_path = f"C:\\Users\\sss\\Desktop\\images\\{idx + 1}_{timestamp}.jpg"
                            current_image.save(save_path, "JPEG")
                            print(f"窗口截图已保存至: {save_path}")

                            # 更新前一张图片
                            previous_images[hwnd] = current_image
                        else:
                            print(f"窗口 {title} 的内容未变化，跳过保存。")

                    except Exception as e:
                        print(f"无法截取窗口 {title}，错误: {e}")

                # 间隔 1 秒
                time.sleep(1)

        except KeyboardInterrupt:
            print("截图循环已终止。")
