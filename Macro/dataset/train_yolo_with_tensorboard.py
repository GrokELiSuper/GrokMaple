# -*- coding: utf-8 -*-
from ultralytics import YOLO
import os
from pathlib import Path
import subprocess
import threading
import glob
import time
import torch

# TensorBoard ?? ?? (?????)
def launch_tensorboard(logdir='runs/detect', port=6007):
    def _run():
        subprocess.run(["tensorboard", "--logdir", logdir, f"--port={port}"])
    threading.Thread(target=_run, daemon=True).start()
    print(f"TensorBoard launched at http://localhost:{port}")

# ?? ??
'''
def train_yolo():
    model = YOLO("yolov8x.pt")  # ?? yolov8n.pt, yolov8m.pt, yolov8l.pt ? ?? ??
    model.train(
        data="data.yaml",         # data.yaml ??
        epochs=200,               # ?? ? (??? 100 ??, ? ????? ?? ??)
        imgsz=640,               # ??? ?? ?? (YOLO? ?????? resize)
        batch=32,                # ?? GPU? ?? ?? ??? (??? ??: 32 * 4 = 128)
        name="nunu_elna_exp1",   # ?? ?? ?? ?? (runs/detect/nunu_elna_exp1)
        project='runs/detect',   # runs/detect/exp1 ?? ? ?? ??? ??
        device=[0,1,2,3],        # 4?? V100 GPU ??
        workers=16,              # ??? ?? ?? ?? ? (4 GPU? ??)
        verbose=True
    )
'''

# fine-tuning ?? ?? (player_dot ??)
def fine_tune_yolo():
    model = YOLO("yolov8l.pt")  # ?? yolov8n.pt, yolov8m.pt, yolov8l.pt ? ?? ??
    model.train(
        data="player_dot_finetune.yaml",
        epochs=50,
        imgsz=320,
        batch=32,                # ??? ??: 32 * 4 = 128
        name="player_dot_finetune",
        project="runs/fine_tune",
        device=[0,1,2,3],        # 4?? V100 GPU ??
        workers=16,              # ??? ?? ?? ?? ?
        verbose=True
    )

# ?? ?? (?? ???)
def unified_predict_pipeline(original_weights="runs/detect/nunu_elna_exp1/weights/best.pt",
                            finetuned_weights="runs/fine_tune/player_dot_finetune/weights/best.pt",
                            source="sample.jpg",
                            conf_orig=0.18, conf_fine=0.16):
    yolo_orig = YOLO(original_weights)
    yolo_fine = YOLO(finetuned_weights)

    result_orig = yolo_orig.predict(source=source, conf=conf_orig, save=False)[0]
    result_fine = yolo_fine.predict(source=source, conf=conf_fine, save=False)[0]

    merged_boxes = result_orig.boxes.data.tolist() + result_fine.boxes.data.tolist()
    print(f"?? ? {len(merged_boxes)}? ??? (orig: {len(result_orig.boxes)}, fine: {len(result_fine.boxes)})")
    return merged_boxes

# ?? ?? ??? ??
def unified_multiple_image_predict(
        folder="predict_images",
        original_weights="runs/detect/nunu_elna_exp1/weights/best.pt",
        finetuned_weights="runs/fine_tune/player_dot_finetune/weights/best.pt",
        conf_orig=0.18,
        conf_fine=0.16,
        tag="merged"
):
    yolo_orig = YOLO(original_weights)
    yolo_fine = YOLO(finetuned_weights)
    image_list = glob.glob(f"{folder}/*.jpg") + glob.glob(f"{folder}/*.png")

    for img_path in image_list:
        result_orig = yolo_orig.predict(source=img_path, conf=conf_orig, save=False)[0]
        result_fine = yolo_fine.predict(source=img_path, conf=conf_fine, save=False)[0]

        merged_boxes = result_orig.boxes.data.tolist() + result_fine.boxes.data.tolist()
        print(f"{img_path} ?? ? {len(merged_boxes)}? ???")

"""
# ??? ?? ?? (Commented out due to missing dxcam dependency)
def real_time_capture_predict(weights_path="runs/detect/nunu_elna_exp1/weights/best.pt",
                             conf_thres=0.18):
    model = YOLO(weights_path)
    camera = dxcam.create(output_color="BGR")
    camera.start(target_fps=10)

    region = (0, 0, 1280, 720)  # QHD ?? ?? 1/4

    while True:
        frame = camera.get_latest_frame(region=region)
        if frame is None:
            continue

        results = model.predict(
            source=frame,
            conf=conf_thres,
            save_crop=True,
            save=False,
            show=False
        )

        for box in results[0].boxes.data:
            cls = int(box[5])
            if model.names[cls] == "projectile_attack":
                print("?? ???!")

        time.sleep(0.5)
"""

if __name__ == '__main__':
    # ?? ???? ?? ? TensorBoard ??
    logdir = Path('runs/detect')
    if not logdir.exists():
        logdir.mkdir(parents=True, exist_ok=True)

    # Only launch TensorBoard on rank 0 to avoid multiple instances
    if 'LOCAL_RANK' not in os.environ or int(os.environ['LOCAL_RANK']) == 0:
        launch_tensorboard(logdir=str(logdir))

    # train_yolo()  # ?? ??? ?? ?? ??
    fine_tune_yolo()  # fine-tuning ??

    # ????? ???? YOLO? ?? ??? ????
    # ?? ??(???, ?? ?)? DQN ??? ??????