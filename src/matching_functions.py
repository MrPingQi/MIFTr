import cv2
import numpy as np


def draw_matches(img1, img2, keypoints1, keypoints2, color=(0, 255, 0)):
    if keypoints1 is not None and keypoints2 is not None:
        n = min(len(keypoints1), len(keypoints2))
        kp1 = [cv2.KeyPoint(kp[0], kp[1], 1) for kp in keypoints1[:n]]
        kp2 = [cv2.KeyPoint(kp[0], kp[1], 1) for kp in keypoints2[:n]]
        matches = [cv2.DMatch(i, i, 1) for i in range(n)]
    else:
        kp1, kp2, matches = [], [], []

    img1, img2 = _to_bgr(img1), _to_bgr(img2)
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    vertical_flag = (h1 + h2) < 0.7 * (w1 + w2)

    if vertical_flag:
        return draw_matches_vertical(img1, img2, kp1, kp2, matches, color=color)
    else:
        return cv2.drawMatches(img1, kp1, img2, kp2, matches, None, matchColor=color)


def draw_matches_vertical(img1, img2, kp1, kp2, dmatches, color=(0, 255, 0), radius=3, thickness=1):
    """把两图上下叠起并画匹配（手动竖拼）"""

    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    H, W = h1 + h2, max(w1, w2)

    # 画布（黑底）
    out = np.zeros((H, W, 3), dtype=img1.dtype)

    # 将两幅图贴到画布上（不足宽度的右侧留黑）
    out[0:h1, 0:w1] = img1
    out[h1:h1+h2, 0:w2] = img2
    if len(dmatches) == 0:
        return out

    # 画匹配连线与关键点
    for m in dmatches:
        pt1 = tuple(np.round(kp1[m.queryIdx].pt).astype(int))
        pt2 = tuple(np.round(kp2[m.trainIdx].pt).astype(int))

        # 边界裁剪（避免越界报错）
        if not (0 <= pt1[0] < W and 0 <= pt1[1] < h1):
            continue
        if not (0 <= pt2[0] < W and 0 <= pt2[1] < h2):
            continue

        pt2_shift = (pt2[0], pt2[1] + h1)
        cv2.circle(out, pt1, radius, color, -1)
        cv2.circle(out, pt2_shift, radius, color, -1)
        cv2.line(out, pt1, pt2_shift, color, thickness)

    return out


def _to_bgr(img):
    """确保是三通道 BGR（灰度图转 BGR）"""
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img


# 单图自动尺寸/尺度限制
def scale_limit(image=None,
                edge_max=1600,
                edge_min=512,
                edge_int=1,
                force_scale=1,
                force_size=None,
                pad_flag=True,
                square_flag=False
                ):
    if image is not None:
        # 判断输入是否含通道维，不改变数据结构
        gray_flag = True if image.ndim < 3 else False

        # (h0,w0): 图像原始尺寸， (h1,w1): 图像缩放尺寸， (h2,w2): 图像最终尺寸
        h0, w0 = image.shape[:2]

        # 非强制图像尺寸
        if force_size is None:
            # 强制图像缩放比例
            if isinstance(force_scale, (int, float)):
                force_scale = (force_scale, force_scale)
            h1, w1 = h0 * force_scale[0], w0 * force_scale[1]

            # 分别计算满足 edge_max 和 edge_min 的缩放比
            # 如果图像的短边 已经 ≥ edge_min，就不需要再放大
            # 如果长边 已经 ≤ edge_max，也无需缩小
            # 只有当 长边 > edge_max 或 短边 < edge_min 时才需要缩放
            # 若两者冲突（即为了满足 edge_min 会导致长边 > edge_max），必须以 edge_max 为准，牺牲 edge_min
            long_edge, short_edge, scale = max(h1, w1), min(h1, w1), 1.0
            if long_edge > edge_max:
                scale = edge_max / long_edge
            elif short_edge < edge_min:
                scale = edge_min / short_edge
                if max(h1, w1) * scale > edge_max:
                    scale = edge_max / long_edge

            # 应用缩放
            h1, w1 = h1 * scale, w1 * scale

            # 图像尺寸整数倍数（算法要求）
            h2, w2 = map(lambda x: int(x // edge_int * edge_int), [h1, w1])

            # 强制图像正方形
            if square_flag:
                h2, w2 = max(h2, w2), max(h2, w2)

            h1, w1 = h2, w2
            if pad_flag:
                ratio = min(h2 / h1, w2 / w1)
                h1, w1 = int(h1 * ratio), int(w1 * ratio)

        # 强制图像尺寸
        else:
            if isinstance(force_size, int):
                force_size = (force_size, force_size)
            h2, w2 = force_size[0], force_size[1]
            h1, w1 = h2, w2
            if pad_flag:
                scale = min(h2 / h0, w2 / w0)
                h1, w1 = int(h0 * scale), int(w0 * scale)

        # 图像缩放
        image = cv2.resize(image, (w1, h1))
        scale = np.array([w0 / w1, h0 / h1], dtype=np.float32)

        # 图像Padding
        mask = None
        if pad_flag:
            if image.ndim < 3:
                image = image[..., None]
            image = np.pad(image, pad_width=((0, h2 - h1), (0, w2 - w1), (0, 0)), mode='constant')
            mask = np.zeros((h2, w2), dtype=bool)
            mask[:h1, :w1] = True

        if gray_flag and image.ndim >= 3:
            image = np.squeeze(image, axis=-1)

        return image, scale, mask


# 双图自动尺寸/尺度限制
def scale_limit2(image0=None, image1=None,
                 edge_max=1600,
                 edge_min=1,
                 edge_int=1,
                 force_scale=1,
                 force_size=None,
                 pad_flag=True,
                 square_flag=False
                 ):
    if image0 is not None and image1 is not None:
        # 判断输入是否含通道维，不改变数据结构
        gray_flag0 = True if image0.ndim < 3 else False
        gray_flag1 = True if image1.ndim < 3 else False

        # (h0,w0): 图像原始尺寸， (h1,w1): 图像缩放尺寸， (h2,w2): 图像最终尺寸
        h00, w00 = image0.shape[:2]
        h10, w10 = image1.shape[:2]

        # 非强制图像尺寸
        if force_size is None:
            # 强制图像缩放比例
            force_scale = 1 if force_scale is None else force_scale
            if isinstance(force_scale, (int, float)):
                force_scale = (force_scale, force_scale)
            h01, w01 = h00 * force_scale[0], w00 * force_scale[1]
            h11, w11 = h10 * force_scale[0], w10 * force_scale[1]

            # 分别计算满足 edge_max 和 edge_min 的缩放比
            # 如果图像的短边 已经 ≥ edge_min，就不需要再放大
            # 如果长边 已经 ≤ edge_max，也无需缩小
            # 只有当 长边 > edge_max 或 短边 < edge_min 时才需要缩放
            # 若两者冲突（即为了满足 edge_min 会导致长边 > edge_max），必须以 edge_max 为准，牺牲 edge_min
            long0, short0, scale0 = max(h01, w01), min(h01, w01), 1.0
            long1, short1, scale1 = max(h11, w11), min(h11, w11), 1.0

            # 计算缩放比例（图像0）
            if long0 > edge_max:
                scale0 = edge_max / long0
            elif short0 < edge_min:
                scale0 = edge_min / short0
                if short0 * scale0 > edge_max:
                    scale0 = edge_max / long0

            # 计算缩放比例（图像1）
            if long1 > edge_max:
                scale1 = edge_max / long1
            elif short1 < edge_min:
                scale1 = edge_min / short1
                if short1 * scale1 > edge_max:
                    scale1 = edge_max / long1

            # 应用缩放
            h01, w01 = h01 * scale0, w01 * scale0
            h11, w11 = h11 * scale1, w11 * scale1

            # 图像相同尺寸（算法要求）
            h2, w2 = max(h01, h11), max(w01, w11)

            # 图像尺寸整数倍数（算法要求）
            edge_int = 1 if edge_int is None else edge_int
            h2, w2 = map(lambda x: int(x // edge_int * edge_int), [h2, w2])

            # 强制图像正方形
            if square_flag:
                h2, w2 = max(h2, w2), max(h2, w2)

            if pad_flag:
                # scale = min(h2 / h01, h2 / h11, w2 / w01, w2 / w11)
                # h01, w01, h11, w11 = int(h01 * scale), int(w01 * scale), int(h11 * scale), int(w11 * scale)
                scale0 = min(h2 / h01, w2 / w01)
                scale1 = min(h2 / h11, w2 / w11)
                h01, w01 = int(h01 * scale0), int(w01 * scale0)
                h11, w11 = int(h11 * scale1), int(w11 * scale1)
            else:
                h01, w01, h11, w11 = h2, w2, h2, w2

        # 强制图像尺寸
        else:
            if isinstance(force_size, int):
                force_size = (force_size, force_size)
            h2, w2 = force_size[0], force_size[1]
            if pad_flag:
                scale = min(h2 / h00, h2 / h10, w2 / w00, w2 / w10)
                h01, w01, h11, w11 = int(h00 * scale), int(w00 * scale), int(h10 * scale), int(w10 * scale)
            else:
                h01, w01, h11, w11 = h2, w2, h2, w2

        # 图像缩放
        image0 = cv2.resize(image0, (w01, h01))
        image1 = cv2.resize(image1, (w11, h11))
        scale0 = np.array([w00 / w01, h00 / h01], dtype=np.float32)
        scale1 = np.array([w10 / w11, h10 / h11], dtype=np.float32)

        # 图像Padding
        if pad_flag:
            if image0.ndim < 3:
                image0 = image0[..., None]
            if image1.ndim < 3:
                image1 = image1[..., None]
            image0 = np.pad(image0, pad_width=((0, h2 - h01), (0, w2 - w01), (0, 0)), mode='constant')
            image1 = np.pad(image1, pad_width=((0, h2 - h11), (0, w2 - w11), (0, 0)), mode='constant')
            mask0, mask1 = np.zeros((h2, w2), dtype=bool), np.zeros((h2, w2), dtype=bool)
            mask0[:h01, :w01] = True
            mask1[:h11, :w11] = True
        else:
            mask0, mask1 = np.ones((h2, w2), dtype=bool), np.ones((h2, w2), dtype=bool)

        if gray_flag0 and image0.ndim >= 3:
            image0 = np.squeeze(image0, axis=-1)
        if gray_flag1 and image1.ndim >= 3:
            image1 = np.squeeze(image1, axis=-1)

        if not gray_flag0 and image0.ndim < 3:
            image0 = image0[..., None]
        if not gray_flag1 and image1.ndim < 3:
            image1 = image1[..., None]

        return image0, image1, scale0, scale1, mask0, mask1
