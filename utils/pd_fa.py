import numpy as np
import cv2
from skimage import measure

class PD_FA():
    def __init__(self,):
        super(PD_FA, self).__init__()
        self.image_area_total = []
        self.image_area_match = []
        self.dismatch_pixel = 0
        self.all_pixel = 0
        self.PD = 0
        self.target= 0

    @staticmethod
    def _sigmoid(x):
        # 防溢出 Sigmoid
        x = np.clip(x, -50, 50)
        return 1.0 / (1.0 + np.exp(-x))

    @staticmethod
    def _to_prob_map(preds: np.ndarray) -> np.ndarray:
        """把模型输出变成 0-1 概率并做稳健归一化。"""
        preds = np.asarray(preds)

        # 若像 logits（<0 或 >1），先过 sigmoid；否则假定已是概率/分数
        if preds.min() < 0 or preds.max() > 1:
            preds = PD_FA._sigmoid(preds)

        m = np.max(preds)
        if not np.isfinite(m) or m <= 0:
            # 全零/全常数：直接裁剪到[0,1]，避免 /0
            preds = np.clip(preds, 0.0, 1.0)
        else:
            preds = preds / (m + 1e-8)

        # 清理任何潜在 NaN/Inf
        preds = np.nan_to_num(preds, nan=0.0, posinf=1.0, neginf=0.0)
        return preds

    @staticmethod
    def _to_binary_mask(label) -> np.ndarray:
        """把 GT 统一成 {0,1} 二值图"""
        label = np.asarray(label)
        if label.dtype != np.uint8:
            label = label.astype(np.uint8)
        if label.max() > 1:
            label = (label > 0).astype(np.uint8)
        return label

    def update(self, preds, labels, size, thr: float = 0.5, match_thresh: float = 3.0):
        """
        preds: HxW 或 1xHxW 的概率/分数/Logits
        labels: HxW (0/1 或 0/255)
        size: (H, W)
        thr:   分割阈值
        match_thresh: 质心匹配半径（像素）
        """
        # 1) 稳健概率图
        preds = self._to_prob_map(preds)
        if preds.ndim == 3 and preds.shape[0] == 1:
            preds = preds[0]
        preds_bin = (preds >= thr).astype(np.uint8)

        # 2) 二值化 GT
        labelss = self._to_binary_mask(labels)

        # 3) 连通域
        # connectivity=2 在 2D 下是 8 邻接
        lab_pred  = measure.label(preds_bin, connectivity=2)
        props_pred = measure.regionprops(lab_pred)

        lab_gt   = measure.label(labelss, connectivity=2)
        props_gt = measure.regionprops(lab_gt)

        # 累计目标数（用于 PD）
        self.target += len(props_gt)

        # 清空本轮缓存
        self.image_area_total = []
        self.image_area_match = []
        self.distance_match   = []
        self.dismatch         = []

        # 记录所有预测连通域的面积
        for p in props_pred:
            self.image_area_total.append(int(p.area))

        # 4) 质心匹配（每个 GT 只匹配一个最近的预测；预测也不能被复用）
        used_pred_idx = set()
        for g in props_gt:
            cy, cx = g.centroid  # 注意 regionprops 返回 (row, col)
            best_idx = -1
            best_dist = np.inf
            for idx, p in enumerate(props_pred):
                if idx in used_pred_idx:
                    continue
                py, px = p.centroid
                d = np.hypot(py - cy, px - cx)
                if d < best_dist:
                    best_dist = d
                    best_idx = idx
            if best_dist <= match_thresh and best_idx >= 0:
                used_pred_idx.add(best_idx)
                self.distance_match.append(best_dist)
                self.image_area_match.append(int(props_pred[best_idx].area))

        # 5) 未匹配的预测像素 → 记为误报像素
        # （用集合差，而不是在循环里 del，避免索引错乱）
        matched_areas_set = set(self.image_area_match)
        # 如果同面积出现多次，集合会丢重；改用计数更稳健：
        from collections import Counter
        total_cnt   = Counter(self.image_area_total)
        matched_cnt = Counter(self.image_area_match)
        dismatch_pixel = 0
        for area, cnt in total_cnt.items():
            k = cnt - matched_cnt.get(area, 0)
            if k > 0:
                dismatch_pixel += area * k

        self.dismatch_pixel += dismatch_pixel

        # 6) 累计图像总像素（用于 FA=误报像素/总像素）
        H, W = int(size[0]), int(size[1])
        self.all_pixel += H * W

        # 7) PD（命中目标数）
        self.PD += len(self.distance_match)

    def get(self):
        # 防止除零
        Final_FA = (self.dismatch_pixel / self.all_pixel) if self.all_pixel > 0 else 0.0
        Final_PD = (self.PD / self.target) if self.target > 0 else 0.0
        return float(Final_PD), float(Final_FA)
