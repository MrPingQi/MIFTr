import warnings
warnings.filterwarnings("ignore")
import torch
import os
import cv2
import matplotlib.pyplot as plt
import imageio.v2 as imageio
from tqdm import tqdm

from src.matching_functions import draw_matches, scale_limit2
from miftr_demo import MIFTr, cfg


if __name__ == "__main__":
    # If matching is difficult, you can try lowering the threshold.
    config = cfg
    config["match_coarse"]["border_rm"] = 2  # default: 2
    config["match_coarse"]["thr"] = 0.1  # default: 0.2
    config["fine"]["thr"] = 0.1  # default: 0.1

    half_precision = True  # default: False

    # If image sizes are too small, such as smaller than 384×384, use dgim_256.ckpt; otherwise, use dgim_512.ckpt.
    # Choice: MIFTr full / light model
    matcher = MIFTr(config=config, model_type='full', pretrained='weights/miftr_full_tune.ckpt')
    # matcher = MIFTr(config=config, model_type='light', pretrained='weights/miftr_light_tune.ckpt')
    
    matcher = matcher.eval().cuda()
    if half_precision:
        matcher = matcher.half()

    # If matching is difficult, the input images can be resized proportionally,
    # and the output matches will be automatically resized back.
    # scale_ratio0 = 1.0  # default: 1.0
    # scale_ratio1 = 1.0  # default: 1.0

    save_path = "./outputs/"
    folder_path = "./sample/"
    image_names = sorted(os.listdir(folder_path))
    for i in tqdm(range(0, len(image_names), 2)):
        image0_bgr = cv2.imread(folder_path + "/" + image_names[i])
        image1_bgr = cv2.imread(folder_path + "/" + image_names[i + 1])
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
            mkpts0 = batch["mkpts0_f"].cpu().numpy()
            mkpts1 = batch["mkpts1_f"].cpu().numpy()
        torch.cuda.empty_cache()

        if len(mkpts0) >= 4:
            homography, mask = cv2.findHomography(mkpts1, mkpts0, cv2.USAC_MAGSAC, 5.0)
            mkpts0 = mkpts0[mask.flatten() == 1] * scale0
            mkpts1 = mkpts1[mask.flatten() == 1] * scale1

        matches_visual = draw_matches(image0_rgb, image1_rgb, mkpts0, mkpts1)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        imageio.imsave(save_path+f"{i//2+1:04d}_matches_correct_{len(mkpts0):04d}.png", matches_visual)

        # plt.figure()
        # plt.imshow(matches_visual, "gray")
        # plt.title(f"{len(mkpts0)} matches")
        # plt.axis("off")
        # plt.show()
