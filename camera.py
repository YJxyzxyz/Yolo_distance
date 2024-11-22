# -- coding: utf-8 --
import argparse
import time
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
from models.experimental import attempt_load
from utils.dataloaders import LoadStreams  # 读取视频流
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

# 直接在代码中指定参数
class Options:
    def __init__(self):
        self.source = '0'  # 使用摄像头，数字 '0' 表示默认摄像头
        self.weights = 'yolov5s.pt'   # 模型权重路径
        self.img_size = 640            # 图像尺寸
        self.project = 'runs/detect'   # 项目目录
        self.name = 'exp'              # 保存文件夹名称
        self.exist_ok = True            # 允许存在相同文件名
        self.save_txt = False           # 保存检测结果到txt
        self.save_csv = False            # 保存结果到CSV
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
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)  # 使用 LoadStreams 读取摄像头
    else:
        save_img = True
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time()

    for paths, img, im0s, vid_cap, s in dataset:
        path = paths[0]  # 获取第一个路径，假设我们只处理一个流
        print(f"Processing: {path}")

        # im0s 应该是列表，获取实际的图像帧
        im0 = im0s[0] if isinstance(im0s, list) else im0s  # 获取第一个图像帧

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
            p, s, frame = path, '', im0  # 使用 im0 作为帧

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

                    if save_img or view_img:  # Add bbox to image
                        label = f'{names[int(cls)]} {conf:.2f} '
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)

            # Print time (inference + NMS)
            print(f'{s}Done. ({t2 - t1:.3f}s)')

            # Stream results
            if view_img:
                cv2.imshow('Camera Feed', im0)  # 显示摄像头图像
                # 等待1毫秒并检查按键
                if cv2.waitKey(1) & 0xFF == ord('q'):  # 如果按下 'q' 键
                    print("Exiting...")
                    break  # 退出循环

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
    cv2.destroyAllWindows()  # 关闭所有 OpenCV 窗口

if __name__ == "__main__":
    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()  # 如果没有更新，直接检测

