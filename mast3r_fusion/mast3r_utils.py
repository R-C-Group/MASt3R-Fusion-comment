import PIL
import numpy as np
import torch
"""
mast3r_utils.py — MASt3R 模型加载与推理封装模块

功能描述:
    封装了对 MASt3R 深度学习模型的所有调用，是连接核心模型与 SLAM 系统的桥梁。
    
    主要功能:
    1. 模型加载:
       - load_mast3r(): 加载预训练的 MASt3R 模型
       - load_retriever(): 加载图像检索模型
    2. 推理:
       - mast3r_inference_mono(): 单目推理（自身配对，初始化用）
       - mast3r_symmetric_inference(): 两帧对称推理（编码 + 解码）
       - mast3r_decode_symmetric_batch(): 批量对称解码
    3. 匹配:
       - mast3r_match_symmetric(): 对称匹配（解码+匹配+质量评估）
       - mast3r_match_asymmetric(): 非对称匹配（用于帧跟踪）
    4. 工具:
       - resize_img(): 图像缩放到模型所需尺寸
       - downsample_feats(): 特征降采样
    
    MASt3R 输出（每对图像）:
    - X (pts3d): 3D 点云（相机坐标系）
    - C (conf): 3D 点的置信度
    - D (desc): 用于匹配的密集描述子
    - Q (desc_conf): 描述子置信度
"""

import functools

import einops

import mast3r.utils.path_to_dust3r  # noqa
from dust3r.utils.image import ImgNorm
from mast3r.model import AsymmetricMASt3R
from mast3r_fusion.retrieval_database import RetrievalDatabase
from mast3r_fusion.config import config
import mast3r_fusion.matching as matching


def load_mast3r(path=None, device="cuda"):
    # 主模型负责输出点图、描述子和置信度，是整个视觉前端的“特征工厂”。
    weights_path = (
        "checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"
        if path is None
        else path
    )
    model = AsymmetricMASt3R.from_pretrained(weights_path).to(device)
    return model


def load_retriever(mast3r_model, retriever_path=None, device="cuda"):
    # 检索器和主模型共用 backbone，作用不是做精确配准，
    # 而是快速提出“哪些历史关键帧值得尝试建立约束”。
    retriever_path = (
        "checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_trainingfree.pth"
        if retriever_path is None
        else retriever_path
    )
    retriever = RetrievalDatabase(retriever_path, backbone=mast3r_model, device=device)
    return retriever


@torch.inference_mode
def decoder(model, feat1, feat2, pos1, pos2, shape1, shape2):
    # 编码器特征在 `frame.feat/pos` 中缓存后，可被反复送入 decoder，
    # 避免每次匹配都重新跑整张图的编码器。
    dec1, dec2 = model._decoder(feat1, pos1, feat2, pos2)
    with torch.amp.autocast(enabled=False, device_type="cuda"):
        res1 = model._downstream_head(1, [tok.float() for tok in dec1], shape1)
        res2 = model._downstream_head(2, [tok.float() for tok in dec2], shape2)
    return res1, res2


def downsample(X, C, D, Q):
    # 数据集级别的降采样会同时作用到几何、置信度和描述子，
    # 这样整个系统都在同一分辨率上工作，避免索引错位。
    downsample = config["dataset"]["img_downsample"]
    if downsample > 1:
        # C and Q: (...xHxW)
        # X and D: (...xHxWxF)
        X = X[..., ::downsample, ::downsample, :].contiguous()
        C = C[..., ::downsample, ::downsample].contiguous()
        D = D[..., ::downsample, ::downsample, :].contiguous()
        Q = Q[..., ::downsample, ::downsample].contiguous()
    return X, C, D, Q


@torch.inference_mode
def mast3r_symmetric_inference(model, frame_i, frame_j):
    # 对称推理会跑两次 decoder：
    # i 看 j、j 看 i。这样后续能同时得到双向匹配质量。
    if frame_i.feat is None:
        frame_i.feat, frame_i.pos, _ = model._encode_image(
            frame_i.img, frame_i.img_true_shape
        )
    if frame_j.feat is None:
        frame_j.feat, frame_j.pos, _ = model._encode_image(
            frame_j.img, frame_j.img_true_shape
        )

    feat1, feat2 = frame_i.feat, frame_j.feat
    pos1, pos2 = frame_i.pos, frame_j.pos
    shape1, shape2 = frame_i.img_true_shape, frame_j.img_true_shape

    res11, res21 = decoder(model, feat1, feat2, pos1, pos2, shape1, shape2)
    res22, res12 = decoder(model, feat2, feat1, pos2, pos1, shape2, shape1)
    res = [res11, res21, res22, res12]
    X, C, D, Q = zip(
        *[(r["pts3d"][0], r["conf"][0], r["desc"][0], r["desc_conf"][0]) for r in res]
    )
    # 4xhxwxc
    X, C, D, Q = torch.stack(X), torch.stack(C), torch.stack(D), torch.stack(Q)
    X, C, D, Q = downsample(X, C, D, Q)
    return X, C, D, Q


# NOTE: Assumes img shape the same
@torch.inference_mode
def mast3r_decode_symmetric_batch(
    model, feat_i, pos_i, feat_j, pos_j, shape_i, shape_j
):
    B = feat_i.shape[0]
    X, C, D, Q = [], [], [], []
    for b in range(B):
        feat1 = feat_i[b][None]
        feat2 = feat_j[b][None]
        pos1 = pos_i[b][None]
        pos2 = pos_j[b][None]
        res11, res21 = decoder(model, feat1, feat2, pos1, pos2, shape_i[b], shape_j[b])
        res22, res12 = decoder(model, feat2, feat1, pos2, pos1, shape_j[b], shape_i[b])
        res = [res11, res21, res22, res12]
        Xb, Cb, Db, Qb = zip(
            *[
                (r["pts3d"][0], r["conf"][0], r["desc"][0], r["desc_conf"][0])
                for r in res
            ]
        )
        X.append(torch.stack(Xb, dim=0))
        C.append(torch.stack(Cb, dim=0))
        D.append(torch.stack(Db, dim=0))
        Q.append(torch.stack(Qb, dim=0))

    X, C, D, Q = (
        torch.stack(X, dim=1),
        torch.stack(C, dim=1),
        torch.stack(D, dim=1),
        torch.stack(Q, dim=1),
    )
    X, C, D, Q = downsample(X, C, D, Q)
    return X, C, D, Q


@torch.inference_mode
def mast3r_inference_mono(model, frame):
    # 初始化阶段用“自己和自己配对”的方式，让网络直接给出单帧点图。
    # 这不是严格意义的单目深度网络，而是复用 MASt3R 的双目接口。
    if frame.feat is None:
        frame.feat, frame.pos, _ = model._encode_image(frame.img, frame.img_true_shape)

    feat = frame.feat
    pos = frame.pos
    shape = frame.img_true_shape

    res11, res21 = decoder(model, feat, feat, pos, pos, shape, shape)
    res = [res11, res21]
    X, C, D, Q = zip(
        *[(r["pts3d"][0], r["conf"][0], r["desc"][0], r["desc_conf"][0]) for r in res]
    )
    # 4xhxwxc
    X, C, D, Q = torch.stack(X), torch.stack(C), torch.stack(D), torch.stack(Q)
    X, C, D, Q = downsample(X, C, D, Q)

    Xii, Xji = einops.rearrange(X, "b h w c -> b (h w) c")
    Cii, Cji = einops.rearrange(C, "b h w -> b (h w) 1")

    return Xii, Cii


def mast3r_match_symmetric(model, feat_i, pos_i, feat_j, pos_j, shape_i, shape_j, subpixel_factor = 1):
    X, C, D, Q = mast3r_decode_symmetric_batch(
        model, feat_i, pos_i, feat_j, pos_j, shape_i, shape_j
    )

    # Ordering 4xbxhxwxc
    b = X.shape[1]

    Xii, Xji, Xjj, Xij = X[0], X[1], X[2], X[3]
    Dii, Dji, Djj, Dij = D[0], D[1], D[2], D[3]
    Qii, Qji, Qjj, Qij = Q[0], Q[1], Q[2], Q[3]

    # 通过拼接的方式把 i->j 与 j->i 两个方向打包成一次批处理匹配，
    # 后面再拆回来，减少重复代码路径。
    X11 = torch.cat((Xii, Xjj), dim=0)
    X21 = torch.cat((Xji, Xij), dim=0)
    D11 = torch.cat((Dii, Djj), dim=0)
    D21 = torch.cat((Dji, Dij), dim=0)

    # tic()
    idx_1_to_2, valid_match_2 = matching.match(X11, X21, D11, D21,subpixel_factor=subpixel_factor)
    # toc("Match")

    # TODO: Avoid this
    match_b = X11.shape[0] // 2
    idx_i2j = idx_1_to_2[:match_b]
    idx_j2i = idx_1_to_2[match_b:]
    valid_match_j = valid_match_2[:match_b]
    valid_match_i = valid_match_2[match_b:]

    return (
        idx_i2j,
        idx_j2i,
        valid_match_j,
        valid_match_i,
        Qii.view(b, -1, 1),
        Qjj.view(b, -1, 1),
        Qji.view(b, -1, 1),
        Qij.view(b, -1, 1),
    )


@torch.inference_mode
def mast3r_asymmetric_inference(model, frame_i, frame_j):
    if frame_i.feat is None:
        frame_i.feat, frame_i.pos, _ = model._encode_image(
            frame_i.img, frame_i.img_true_shape
        )
    if frame_j.feat is None:
        frame_j.feat, frame_j.pos, _ = model._encode_image(
            frame_j.img, frame_j.img_true_shape
        )

    feat1, feat2 = frame_i.feat, frame_j.feat
    pos1, pos2 = frame_i.pos, frame_j.pos
    shape1, shape2 = frame_i.img_true_shape, frame_j.img_true_shape

    res11, res21 = decoder(model, feat1, feat2, pos1, pos2, shape1, shape2)
    res = [res11, res21]
    X, C, D, Q = zip(
        *[(r["pts3d"][0], r["conf"][0], r["desc"][0], r["desc_conf"][0]) for r in res]
    )
    # 4xhxwxc
    X, C, D, Q = torch.stack(X), torch.stack(C), torch.stack(D), torch.stack(Q)
    X, C, D, Q = downsample(X, C, D, Q)
    return X, C, D, Q

import matplotlib.pyplot as plt
def mast3r_match_asymmetric(model, frame_i, frame_j, idx_i2j_init=None):
    # 非对称匹配是在线跟踪默认路径：只保留当前帧到关键帧的一侧结果，
    # 比对称版本更轻，适合逐帧运行。
    X, C, D, Q = mast3r_asymmetric_inference(model, frame_i, frame_j)
    
    b, h, w = X.shape[:-1]
    # 2 outputs per inference
    b = b // 2

    Xii, Xji = X[:b], X[b:]
    Cii, Cji = C[:b], C[b:]
    Dii, Dji = D[:b], D[b:]
    Qii, Qji = Q[:b], Q[b:]

    idx_i2j, valid_match_j = matching.match(
        Xii, Xji, Dii, Dji, idx_1_to_2_init=idx_i2j_init
    )

    # 后续模块都按 `(B, H*W, ...)` 的扁平布局访问像素，因此这里统一 reshape。
    Xii, Xji = einops.rearrange(X, "b h w c -> b (h w) c")
    Cii, Cji = einops.rearrange(C, "b h w -> b (h w) 1")
    Dii, Dji = einops.rearrange(D, "b h w c -> b (h w) c")
    Qii, Qji = einops.rearrange(Q, "b h w -> b (h w) 1")

    return idx_i2j, valid_match_j, Xii, Cii, Qii, Xji, Cji, Qji


def _resize_pil_image(img, long_edge_size):
    S = max(img.size)
    if S > long_edge_size:
        interp = PIL.Image.LANCZOS
    elif S <= long_edge_size:
        interp = PIL.Image.BICUBIC
    new_size = tuple(int(round(x * long_edge_size / S)) for x in img.size)
    return img.resize(new_size, interp)


def resize_img(img, size, square_ok=False, return_transformation=False):
    assert size == 224 or size == 512
    # 这里不仅仅是缩放，还隐含了与 MASt3R 训练/预处理流程一致的裁剪规则。
    # 一旦改这里，后面内参、像素坐标和匹配几何都可能要跟着变。
    img = PIL.Image.fromarray(np.uint8(img * 255))
    W1, H1 = img.size
    if size == 224:
        # resize short side to 224 (then crop)
        img = _resize_pil_image(img, round(size * max(W1 / H1, H1 / W1)))
    else:
        # resize long side to 512
        img = _resize_pil_image(img, size)
    W, H = img.size
    cx, cy = W // 2, H // 2
    if size == 224:
        half = min(cx, cy)
        img = img.crop((cx - half, cy - half, cx + half, cy + half))
    else:
        halfw, halfh = ((2 * cx) // 16) * 8, ((2 * cy) // 16) * 8
        if not (square_ok) and W == H:
            halfh = 3 * halfw / 4
        img = img.crop((cx - halfw, cy - halfh, cx + halfw, cy + halfh))

    res = dict(
        img=ImgNorm(img)[None],
        true_shape=np.int32([img.size[::-1]]),
        unnormalized_img=np.asarray(img),
    )
    if return_transformation:
        scale_w = W1 / W
        scale_h = H1 / H
        half_crop_w = (W - img.size[0]) / 2
        half_crop_h = (H - img.size[1]) / 2
        return res, (scale_w, scale_h, half_crop_w, half_crop_h)

    return res
