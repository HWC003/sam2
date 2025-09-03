import numpy as np
import torch
import matplotlib.pyplot as plt
import time
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from hydra import initialize
from hydra.core.global_hydra import GlobalHydra
from pathlib import Path

def main():
    model_cfg_path = Path("sam2/configs/sam2.1")
    model_cfg_file = "sam2.1_hiera_l.yaml"

    GlobalHydra.instance().clear()

    initialize(config_path=str(model_cfg_path), job_name="sam2_image_test", version_base=None)

    image = Image.open('../test_images/3 bowl test angled.jpg')
    
    image = np.array(image.convert("RGB"))

    print("Created SAM image predictor from SAM2 model...")
    start_t = time.time()

    checkpoint = "./checkpoints/sam2.1_hiera_large.pt"

    sam2_model = build_sam2(config_file=model_cfg_file, ckpt_path=checkpoint, device="cuda")

    predictor = SAM2ImagePredictor(sam2_model)

    print("Created SAM image predictor in {:.3f} seconds".format(time.time() - start_t))

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        # print("Predicting with point prompt...")
        # start_t = time.time()

        # predictor.set_image(image)
        # input_point = np.array([[400, 500]])
        # input_label = np.array([1])
        # masks, scores, logits = predictor.predict(
        #     point_coords=input_point, 
        #     point_labels=input_label,
        #     multimask_output=True  # outputs 3 masks with scores.
        # )                          # set to False to output the highest scoring mask
        # print(f"Predicted masks with point prompt in {time.time() - start_t:.3f} seconds")
        # show_masks(image, masks, scores, point_coords=input_point, input_labels=input_label)

        print("Predicting with box prompt...")
        start_t = time.time()

        predictor.set_image(image)
        input_boxes = np.array([
        [100, 500, 550, 900],
        [400, 200, 800, 500],
        [700, 400, 1200, 800],
        ])

        masks, scores, _ = predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_boxes,
        multimask_output=False,
        )

        print(f"Predicted masks with box prompt in {time.time() - start_t:.3f} seconds")
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        for mask in masks:
            show_mask(mask.squeeze(0), plt.gca(), random_color=True)
        for box in input_boxes:
            show_box(box, plt.gca())
        plt.axis('off')
        plt.show()

    GlobalHydra.instance().clear()


"""
Helper functions for visualization
"""

np.random.seed(3)

def show_mask(mask, ax, random_color=False, borders = True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask = mask.astype(np.uint8)
    mask_image =  mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    if borders:
        import cv2
        contours, _ = cv2.findContours(mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # Try to smooth contours
        contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
        mask_image = cv2.drawContours(mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2)
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))

def show_masks(image, masks, scores, point_coords=None, box_coords=None, input_labels=None, borders=True):
    for i, (mask, score) in enumerate(zip(masks, scores)):
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        show_mask(mask, plt.gca(), borders=borders)
        if point_coords is not None:
            assert input_labels is not None
            show_points(point_coords, input_labels, plt.gca())
        if box_coords is not None:
            # boxes
            show_box(box_coords, plt.gca())
        if len(scores) > 1:
            plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
        plt.axis('off')
        plt.show()

if __name__ == "__main__":
    main()

    

