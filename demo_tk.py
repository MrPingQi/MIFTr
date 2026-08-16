import warnings
warnings.filterwarnings("ignore")
import torch
import sys
import cv2
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog

from src.matching_functions import draw_matches, scale_limit2
from miftr_demo import MIFTr, cfg


if __name__ == "__main__":
    # If matching is difficult, you can try lowering the threshold.
    config = cfg
    config["match_coarse"]["border_rm"] = 1  # default: 2
    config["match_coarse"]["thr"] = 0.1  # default: 0.2
    config["fine"]["thr"] = 0.1  # default: 0.1

    half_precision = True  # default: False

    # If image sizes are too small, such as smaller than 384×384, use dgim_256.ckpt; otherwise, use dgim_512.ckpt.
    # Choice: MIFTr full / light model
    matcher = MIFTr(config=config, model_type='full', pretrained='./weights/miftr_full_tune.ckpt')
    # matcher = MIFTr(config=config, model_type='light', pretrained='./weights/miftr_light_tune.ckpt')
    
    matcher = matcher.eval().cuda()
    if half_precision:
        matcher = matcher.half()

    root = tk.Tk()
    root.withdraw()

    image0_path = filedialog.askopenfilename(
        title="Select the Left Image (English-Only Path)",
        filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff *.tif *.webp")]
    )
    if not image0_path:
        sys.exit()

    image1_path = filedialog.askopenfilename(
        title="Select the Right Image (English-Only Path)",
        filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff *.tif *.webp")]
    )
    if not image1_path:
        sys.exit()

    # If matching is difficult, the input images can be resized proportionally,
    # and the output matches will be automatically resized back.
    # scale_ratio0 = 1.0  # default: 1.0
    # scale_ratio1 = 1.0  # default: 1.0

    image0_bgr = cv2.imread(image0_path)
    image1_bgr = cv2.imread(image1_path)
    image0_rgb = cv2.cvtColor(image0_bgr, cv2.COLOR_BGR2RGB)
    image1_rgb = cv2.cvtColor(image1_bgr, cv2.COLOR_BGR2RGB)
    image0_gray = cv2.cvtColor(image0_bgr, cv2.COLOR_BGR2GRAY)
    image1_gray = cv2.cvtColor(image1_bgr, cv2.COLOR_BGR2GRAY)

    image0, image1, scale0, scale1, mask0, mask1 = scale_limit2(image0_gray, image1_gray,
                edge_max=512, edge_min=1, edge_int=16, force_scale=99999, force_size=None, pad_flag=True)

    image0 = torch.from_numpy(image0)[None][None].cuda() / 255
    image1 = torch.from_numpy(image1)[None][None].cuda() / 255
    if half_precision:
        image0 = image0.half()
        image1 = image1.half()

    batch = {"imagec_0": image0, "imagec_1": image1}
    with torch.no_grad():
        matcher(batch)
        matched_points0 = batch["mkpts0_f"].cpu().numpy()
        matched_points1 = batch["mkpts1_f"].cpu().numpy()
    torch.cuda.empty_cache()

    if len(matched_points0) >= 4:
        homography, mask = cv2.findHomography(matched_points1, matched_points0, cv2.USAC_MAGSAC, 5.0)
        matched_points0 = matched_points0[mask.flatten() == 1] * scale0
        matched_points1 = matched_points1[mask.flatten() == 1] * scale1

    matches_visual = draw_matches(image0_rgb, image1_rgb, matched_points0, matched_points1)

    plt.figure()
    plt.imshow(matches_visual, "gray")
    plt.title(f"{len(matched_points0)} matches")
    plt.axis("off")
    plt.show()
