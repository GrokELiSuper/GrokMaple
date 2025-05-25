# -*- coding: utf-8 -*-
from ultralytics import YOLO
import os
from pathlib import Path
import subprocess
import threading
import glob
import time
import torch
import logging
import yaml

try:
    import dxcam
except ImportError:
    print("dxcam not installed. Real-time capture disabled.")
    dxcam = None

# Set up logging to capture training issues
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# TensorBoard ?? ?? (?????)
def launch_tensorboard(logdir='runs/detect', port=6013):
    try:
        def _run():
            subprocess.run(["tensorboard", "--logdir", logdir, f"--port={port}"])
        threading.Thread(target=_run, daemon=True).start()
        logger.info(f"TensorBoard launched at http://localhost:{port}")
    except Exception as e:
        logger.error(f"Failed to launch TensorBoard: {e}")


# ?? ??
def train_yolo():
    try:
        logger.info("Starting train_yolo")
        model = YOLO("yolov8l.pt")  # ??? ?? ?? ??
        model.train(
            data="data.yaml",         # data.yaml ??
            epochs=250,               # ??? ?? ?
            imgsz=640,                # ??? ?? ?? (YOLO? ?????? resize)
            batch=32,                 # ??? ?? ??? ??
            name="nunu_elna_exp1",    # ?? ?? ?? ??
            project='runs/detect',    # runs/detect/exp1 ?? ? ?? ??
            device=[0,1,2,3],         # 4? GPU ??
            workers=16,               # ??? ?? ?? ?? ?
            verbose=True
        )
        logger.info("train_yolo completed successfully")
    except Exception as e:
        logger.error(f"train_yolo failed: {e}")
        raise


# Validate YAML file and dataset paths
def validate_yaml(yaml_path):
    try:
        logger.info(f"Validating YAML file: {yaml_path} (working directory: {os.getcwd()})")
        if not os.path.exists(yaml_path):
            raise FileNotFoundError(f"YAML file not found: {yaml_path}")
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        
        # Use 'path' field if present, otherwise fall back to YAML file's directory
        base_path = data.get('path', os.path.dirname(yaml_path))
        base_path = os.path.abspath(base_path)
        logger.info(f"Using base path: {base_path}")

        for key in ['train', 'val']:
            if key not in data:
                raise ValueError(f"Missing '{key}' in {yaml_path}")
            path = data[key]
            # Resolve relative paths relative to base_path
            abs_path = os.path.abspath(os.path.join(base_path, path))
            if not os.path.exists(abs_path):
                raise FileNotFoundError(f"Dataset path not found: {abs_path} (original: {path})")
        if 'nc' not in data or 'names' not in data:
            raise ValueError(f"Missing 'nc' or 'names' in {yaml_path}")
        logger.info(f"Validated {yaml_path} successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to validate {yaml_path}: {e}")
        return False

# fine-tuning ?? ?? (player_dot ??)
def fine_tune_yolo(weights="runs/detect/nunu_elna_exp1/weights/best.pt"):
    try:
        logger.info("Starting fine_tune_yolo")
        # Check if pre-trained weights exist, fall back to yolov8l.pt if not
        if os.path.exists(weights):
            logger.info(f"Loading pre-trained weights: {weights}")
            model = YOLO(weights)
        else:
            logger.warning(f"Pre-trained weights not found at {weights}, using yolov8l.pt")
            model = YOLO("yolov8l.pt")

        # Validate YAML file
        yaml_path = "player_dot_finetune.yaml"
        if not validate_yaml(yaml_path):
            raise ValueError(f"Invalid YAML configuration: {yaml_path}")

        # Ensure output directory exists
        output_dir = Path("runs/fine_tune")
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Ensured output directory exists: {output_dir}")

        model.train(
            data=yaml_path,
            epochs=250,
            imgsz=320,
            batch=32,
            name="player_dot_finetune",
            project="runs/fine_tune",
            device=[0,1,2,3],
            workers=16,
            verbose=True
        )
        logger.info("fine_tune_yolo completed successfully")
    except Exception as e:
        logger.error(f"fine_tune_yolo failed: {e}")
        raise

# ?? ?? (?? ???)
def unified_predict_pipeline(original_weights="runs/detect/nunu_elna_exp1/weights/best.pt",
                            finetuned_weights="runs/fine_tune/player_dot_finetune/weights/best.pt",
                            source="sample.jpg",
                            conf_orig=0.18, conf_fine=0.16):
    try:
        logger.info(f"Starting unified_predict_pipeline with source: {source}")
        yolo_orig = YOLO(original_weights)
        yolo_fine = YOLO(finetuned_weights)

        result_orig = yolo_orig.predict(source=source, conf=conf_orig, save=False)[0]
        result_fine = yolo_fine.predict(source=source, conf=conf_fine, save=False)[0]

        merged_boxes = result_orig.boxes.data.tolist() + result_fine.boxes.data.tolist()
        logger.info(f"?? ? {len(merged_boxes)}? ??? (orig: {len(result_orig.boxes)}, fine: {len(result_fine.boxes)})")
        return merged_boxes
    except Exception as e:
        logger.error(f"unified_predict_pipeline failed: {e}")
        raise

# ?? ?? ??? ??
def unified_multiple_image_predict(
        folder="predict_images",
        original_weights="runs/detect/nunu_elna_exp1/weights/best.pt",
        finetuned_weights="runs/fine_tune/player_dot_finetune/weights/best.pt",
        conf_orig=0.18,
        conf_fine=0.16,
        tag="merged"
):
    try:
        logger.info(f"Starting unified_multiple_image_predict with folder: {folder}")
        yolo_orig = YOLO(original_weights)
        yolo_fine = YOLO(finetuned_weights)
        image_list = glob.glob(f"{folder}/*.jpg") + glob.glob(f"{folder}/*.png")

        for img_path in image_list:
            result_orig = yolo_orig.predict(source=img_path, conf=conf_orig, save=False)[0]
            result_fine = yolo_fine.predict(source=img_path, conf=conf_fine, save=False)[0]

            merged_boxes = result_orig.boxes.data.tolist() + result_fine.boxes.data.tolist()
            logger.info(f"{img_path} ?? ? {len(merged_boxes)}? ???")
    except Exception as e:
        logger.error(f"unified_multiple_image_predict failed: {e}")
        raise

# ??? ?? ??
def real_time_capture_predict(weights_path="runs/detect/nunu_elna_exp1/weights/best.pt",
                             conf_thres=0.18):
    if dxcam is None:
        logger.error("Cannot run real_time_capture_predict: dxcam not installed")
        return

    try:
        logger.info("Starting real_time_capture_predict")
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
                    logger.info("?? ???!")

            time.sleep(0.5)
    except Exception as e:
        logger.error(f"real_time_capture_predict failed: {e}")
        raise
    finally:
        if 'camera' in locals():
            camera.stop()

if __name__ == '__main__':
    # ?? ???? ?? ? TensorBoard ??
    logdir = Path('runs/detect')
    try:
        if not logdir.exists():
            logdir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created directory: {logdir}")
        # Only launch TensorBoard on rank 0 to avoid multiple instances
        if 'LOCAL_RANK' not in os.environ or int(os.environ['LOCAL_RANK']) == 0:
            launch_tensorboard(logdir=str(logdir))
    except Exception as e:
        logger.error(f"Failed to set up log directory or launch TensorBoard: {e}")
        raise
    #train_yolo()
    fine_tune_yolo()  # Run fine-tuning