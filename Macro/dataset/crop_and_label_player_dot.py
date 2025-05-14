import os
import cv2
import glob

def crop_player_dot_images_and_labels(
    image_dir,
    label_dir,
    output_image_dir,
    output_label_dir,
    target_class_id=0,
    crop_size_margin=5
):
    os.makedirs(output_image_dir, exist_ok=True)
    os.makedirs(output_label_dir, exist_ok=True)

    image_files = glob.glob(os.path.join(image_dir, "*.png")) + \
                  glob.glob(os.path.join(image_dir, "*.jpg"))

    crop_count = 0
    for img_path in image_files:
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(label_dir, f"{img_name}.txt")

        if not os.path.exists(label_path):
            continue

        img = cv2.imread(img_path)
        h, w = img.shape[:2]

        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                cls_id, cx, cy, bw, bh = map(float, parts)
                if int(cls_id) != target_class_id:
                    continue

                px = int((cx - bw / 2) * w)
                py = int((cy - bh / 2) * h)
                pw = int(bw * w)
                ph = int(bh * h)

                x1 = max(px - crop_size_margin, 0)
                y1 = max(py - crop_size_margin, 0)
                x2 = min(px + pw + crop_size_margin, w)
                y2 = min(py + ph + crop_size_margin, h)

                crop = img[y1:y2, x1:x2]
                crop_filename = f"crop_{crop_count}.png"
                crop_path = os.path.join(output_image_dir, crop_filename)
                cv2.imwrite(crop_path, crop)

                # Calculate normalized label (YOLO format: class_id cx cy w h)
                crop_h, crop_w = crop.shape[:2]
                new_cx = 0.5
                new_cy = 0.5
                new_bw = 1.0 - (2 * crop_size_margin / crop_w)
                new_bh = 1.0 - (2 * crop_size_margin / crop_h)

                label_filename = crop_filename.replace(".png", ".txt")
                label_path = os.path.join(output_label_dir, label_filename)
                with open(label_path, "w") as out_label:
                    out_label.write(f"{target_class_id} {new_cx:.6f} {new_cy:.6f} {new_bw:.6f} {new_bh:.6f}\n")

                crop_count += 1

    print(f"✅ Total {crop_count} player_dot images cropped and labeled.")


# ✅ (train)
crop_player_dot_images_and_labels(
    image_dir="elna/images/train",
    label_dir="elna/labels/train",
    output_image_dir="elna/crops/images/train",
    output_label_dir="elna/crops/labels/train"
)

# ✅ (val)
crop_player_dot_images_and_labels(
    image_dir="elna/images/val",
    label_dir="elna/labels/val",
    output_image_dir="elna/crops/images/val",
    output_label_dir="elna/crops/labels/val"
)