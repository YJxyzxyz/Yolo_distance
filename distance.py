# -- coding: utf-8 --
import argparse
import time
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import pandas as pd
import numpy as np
import math
from models.experimental import attempt_load
from utils.dataloaders import LoadStreams, LoadImages
from utils.general import (
    check_img_size, check_requirements, check_imshow,
    non_max_suppression, apply_classifier, scale_boxes,
    xyxy2xywh, strip_optimizer, set_logging, increment_path
)
from utils.plots import plot_one_box
from utils.torch_utils import (
    select_device, load_classifier, time_sync
)
from stereo.dianyuntu_yolo import (
    preprocess, undistortion, getRectifyTransform,
    draw_line, rectifyImage, stereoMatchSGBM,
)
from stereo import stereoconfig_040_2

class DistanceEstimation:
    def __init__(self):
        self.W = 640
        self.H = 480
        self.excel_path = r'./camera_parameters.xlsx'

    def camera_parameters(self, excel_path):
        df_intrinsic = pd.read_excel(excel_path, sheet_name='内参矩阵', header=None)
        df_p = pd.read_excel(excel_path, sheet_name='外参矩阵', header=None)

        print('外参矩阵形状：', df_p.values.shape)
        print('内参矩阵形状：', df_intrinsic.values.shape)

        return df_p.values, df_intrinsic.values

    def object_point_world_position(self, u, v, w, h, p, k):
        u1 = u
        v1 = v + h / 2
        print('关键点坐标：', u1, v1)

        alpha = -(90 + 0) / (2 * math.pi)
        peta = 0
        gama = -90 / (2 * math.pi)

        fx = k[0, 0]
        fy = k[1, 1]
        H = 1
        angle_a = 0
        angle_b = math.atan((v1 - self.H / 2) / fy)
        angle_c = angle_b + angle_a
        print('angle_b', angle_b)

        depth = (H / np.sin(angle_c)) * math.cos(angle_b)
        print('depth', depth)

        k_inv = np.linalg.inv(k)
        p_inv = np.linalg.inv(p)
        point_c = np.array([u1, v1, 1])
        point_c = np.transpose(point_c)
        print('point_c', point_c)
        print('k_inv', k_inv)
        c_position = np.matmul(k_inv, depth * point_c)
        print('c_position', c_position)
        c_position = np.append(c_position, 1)
        c_position = np.transpose(c_position)
        c_position = np.matmul(p_inv, c_position)
        d1 = np.array((c_position[0], c_position[1]), dtype=float)
        return d1

    def distance(self, kuang, xw=5, yw=0.1):
        print('=' * 50)
        print('开始测距')
        p, k = self.camera_parameters(self.excel_path)
        if len(kuang):
            obj_position = []
            u, v, w, h = kuang[1] * self.W, kuang[2] * self.H, kuang[3] * self.W, kuang[4] * self.H
            print('目标框', u, v, w, h)
            d1 = self.object_point_world_position(u, v, w, h, p, k)
        distance = 0
        print('距离', d1)
        if d1[0] <= 0:
            d1[:] = 0
        else:
            distance = math.sqrt(math.pow(d1[0], 2) + math.pow(d1[1], 2))
        return distance, d1

# 直接在代码中指定参数
class Options:
    def __init__(self):
        self.source = 'data/images'  # 指定图像或视频路径
        self.weights = 'yolov5s.pt'   # 模型权重路径
        self.img_size = 640            # 图像尺寸
        self.project = 'runs/detect'   # 项目目录
        self.name = 'exp'              # 保存文件夹名称
        self.exist_ok = False            # 允许存在相同文件名
        self.save_txt = False            # 保存检测结果到txt
        self.save_crop = False           # 保存裁剪后的预测框
        self.nosave = False              # 不保存图像/视频
        self.view_img = True            # 显示结果
        self.update = False             # 是否更新模型
        self.device = ''                # 设备ID (i.e. 0 or cpu)
        self.conf_thres = 0.25          # 置信度阈值
        self.iou_thres = 0.45           # NMS IOU阈值
        self.classes = None              # 类别过滤
        self.agnostic_nms = False       # 类别无关的NMS
        self.augment = False            # 是否使用增强推理
        self.visualize = False          # 可视化特征
        self.save_conf = False          # 保存置信度
        self.line_thickness = 3         # 边界框厚度
        self.hide_labels = False        # 隐藏标签
        self.hide_conf = False          # 隐藏置信度

opt = Options()

def detect(save_img=False):
    num = 210
    source, weights, view_img, save_txt, imgsz = (
        opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    )
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://')
    )

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, device=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    if half:
        model.half()  # to FP16

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        save_img = True
        dataset = LoadImages(source, img_size=imgsz, stride=stride)
        print("img_size:")
        print(imgsz)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    DE = DistanceEstimation()  # Initialize distance estimation

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time()

    for path, img, im0s, vid_cap, s in dataset:
        # 处理逻辑
        print(f"Processing: {path}")
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_sync()
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = time_sync()

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], im0.shape).round()

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    # Measure distance
                    kuang = [int(cls), xywh[0], xywh[1], xywh[2], xywh[3]]
                    distance, d = DE.distance(kuang)

                    if save_img or view_img:  # Add bbox to image
                        label = f'{names[int(cls)]} {conf:.2f} {distance:.2f}m'
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)

            # Print time (inference + NMS)
            print(f'{s}Done. ({t2 - t1:.3f}s)')

            # Stream results
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer

                        fourcc = 'mp4v'  # output video codec
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
                    vid_writer.write(im0)

    print(f'Done. ({time.time() - t0:.3f}s)')

if __name__ == "__main__":
    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()  # 如果没有更新，直接检测
