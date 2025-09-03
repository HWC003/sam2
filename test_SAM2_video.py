import numpy as np
import torch
import matplotlib.pyplot as plt
import time
import os
import cv2
import supervision as sv
from PIL import Image
from sam2.build_sam import build_sam2_video_predictor
from hydra import initialize
from hydra.core.global_hydra import GlobalHydra
from pathlib import Path
from tqdm import tqdm

"""
Params
"""
VIDEO_DIR = "../test_videos/clipped_scooping_medium/"
OUTPUT_VIDEO_PATH = "../test_results/sam2/scooping_tracking_demo.mp4"
SAVE_TRACKING_RESULTS_DIR = "../tracking_results/sam2"

VIDEO_DIR = os.path.expanduser(VIDEO_DIR)
if not os.path.exists(VIDEO_DIR):
    raise ValueError(f"Video directory {VIDEO_DIR} does not exist")

def main():
    model_cfg_path = Path("sam2/configs/sam2.1")
    model_cfg_file = "sam2.1_hiera_l.yaml"

    GlobalHydra.instance().clear()

    initialize(config_path=str(model_cfg_path), job_name="sam2_image_test", version_base=None)

    print("Created SAM video predictor from SAM2 model...")
    start_t = time.time()

    checkpoint = "./checkpoints/sam2.1_hiera_large.pt"

    predictor = build_sam2_video_predictor(config_file=model_cfg_file, ckpt_path=checkpoint, device="cuda")
    
    print("Created SAM video predictor in {:.3f} seconds".format(time.time() - start_t))

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):

        video_dir = VIDEO_DIR

        # scan all the JPEG frame names in this directory
        frame_names = [
            p for p in os.listdir(video_dir)
            if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
        ]
        frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

        print("Initialised video predictor state...")
        start_t = time.time()
        # init video predictor state
        inference_state = predictor.init_state(video_path=video_dir)

        print("Initialised video predictor state in {:.3f} seconds".format(time.time() - start_t))

        print("Predicting with box prompt on frame 0...")
        start_t = time.time()
        ann_frame_idx = 0  # the frame index we interact with
        ann_obj_id = 1  # give a unique id to each object we interact with (it can be any integers)

        box = np.array([347, 351, 435, 404], dtype=np.float32)  # [x0, y0, x1, y1]
        _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=ann_frame_idx,
            obj_id=ann_obj_id,
            box=box,
        )
        print("Predicted masks with box prompt in {:.3f} seconds".format(time.time() - start_t))

        # # show the results on the current (interacted) frame
        # plt.figure(figsize=(9, 6))
        # plt.title(f"frame {ann_frame_idx}")
        # plt.imshow(Image.open(os.path.join(video_dir, frame_names[ann_frame_idx])))
        # show_box(box, plt.gca())
        # show_mask((out_mask_logits[0] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_ids[0])
        # plt.show()

        video_segments = {}
        frame_times = {}   # dict to store per-frame timings
        all_times = []     # or use a list if you only need order

        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
            start_t = time.time()

            # store segmentation results
            video_segments[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }

            elapsed = time.time() - start_t
            frame_times[out_frame_idx] = elapsed
            all_times.append(elapsed)

            print(f"Frame {out_frame_idx}: video predictor took {elapsed:.3f}s")

        # After loop → summary stats
        total_time = sum(all_times)
        avg_time = total_time / len(all_times)
        fps = len(all_times) / total_time

        print(f"\nProcessed {len(all_times)} frames")
        print(f"Total inference time: {total_time:.3f}s")
        print(f"Average per frame: {avg_time:.3f}s ({fps:.2f} FPS)")

        save_dir = SAVE_TRACKING_RESULTS_DIR

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        for frame_idx, segments in video_segments.items():
            img = cv2.imread(os.path.join(VIDEO_DIR, frame_names[frame_idx]))
            
            object_ids = list(segments.keys())
            masks = list(segments.values())
            masks = np.concatenate(masks, axis=0)
            
            detections = sv.Detections(
                xyxy=sv.mask_to_xyxy(masks),  # (n, 4)
                mask=masks, # (n, h, w)
                class_id=np.array(object_ids, dtype=np.int32),
            )
            box_annotator = sv.BoxAnnotator()
            annotated_frame = box_annotator.annotate(scene=img.copy(), detections=detections)
            mask_annotator = sv.MaskAnnotator()
            annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=detections)
            cv2.imwrite(os.path.join(save_dir, f"annotated_frame_{frame_idx:05d}.jpg"), annotated_frame)

        create_video_from_images(SAVE_TRACKING_RESULTS_DIR, OUTPUT_VIDEO_PATH)
    
    GlobalHydra.instance().clear()

"""
Helper Functions for visualization.
"""

def show_mask(mask, ax, obj_id=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=200):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))

def create_video_from_images(image_folder, output_video_path, frame_rate=25):
    # define valid extension
    valid_extensions = [".jpg", ".jpeg", ".JPG", ".JPEG", ".png", ".PNG"]
    
    # get all image files in the folder
    image_files = [f for f in os.listdir(image_folder) 
                   if os.path.splitext(f)[1] in valid_extensions]
    image_files.sort()  # sort the files in alphabetical order
    print(image_files)
    if not image_files:
        raise ValueError("No valid image files found in the specified folder.")
    
    # load the first image to get the dimensions of the video
    first_image_path = os.path.join(image_folder, image_files[0])
    first_image = cv2.imread(first_image_path)
    height, width, _ = first_image.shape
    
    # create a video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # codec for saving the video
    video_writer = cv2.VideoWriter(output_video_path, fourcc, frame_rate, (width, height))
    
    # write each image to the video
    for image_file in tqdm(image_files):
        image_path = os.path.join(image_folder, image_file)
        image = cv2.imread(image_path)
        video_writer.write(image)
    
    # source release
    video_writer.release()
    print(f"Video saved at {output_video_path}")

if __name__ == "__main__":
    main() 
        