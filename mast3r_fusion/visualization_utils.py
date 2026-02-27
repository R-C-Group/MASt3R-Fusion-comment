import functools
import imgui
"""
visualization_utils.py — 可视化辅助工具

功能描述:
    为 3D 可视化提供底层渲染工具函数:
    1. depth2rgb() — 深度图转彩色图（Turbo colormap）
    2. create_frustum() — 创建相机视锥体的线框顶点/颜色
    3. create_frustums() — 批量创建多个关键帧的视锥体
    4. create_frustum_lines() — 创建视锥体的线段索引
    5. render_lines() — 使用 ModernGL 渲染线段

    视锥体形状:
    - 由近平面的 4 个角点和光心构成（共 5 个点，8 条线段）
    - 可以颜色编码置信度或帧类型
"""
import moderngl
import matplotlib
import torch
import numpy as np
from in3d.geometry import LineGeometry


@functools.cache
def get_colormap(colormap):
    colormap = matplotlib.colormaps[colormap]
    return colormap(np.linspace(0, 1, 256))[:, :3]


def depth2rgb(depth, min=None, max=None, colormap="turbo", add_alpha=False, alpha=1.0):
    # depth: HxW
    dmin = np.nanmin(depth) if min is None else min
    dmax = np.nanmax(depth) if max is None else max
    d = (depth - dmin) / np.maximum((dmax - dmin), 1e-8)
    d = np.clip(d * 255, 0, 255).astype(np.int32)
    img = get_colormap(colormap)[d].astype(np.float32)
    if add_alpha:
        img = np.concatenate([img, alpha * np.ones_like(img[..., :1])], axis=-1)
    return np.ascontiguousarray(img)


class Frustums(LineGeometry):
    def __init__(self, program):
        super().__init__()
        self.program = program
        self.lines = []
        self.colors = []
        self.frustum = self.make_frustum(1, 1)

    def make_frustum(self, h, w):
        self.aspect_ratio = float(w / h)
        origin = [0.0, 0.0, 0.0]
        topleft = [-self.aspect_ratio, -1.0, 1.0]
        topright = [self.aspect_ratio, -1.0, 1.0]
        bottomleft = [-self.aspect_ratio, 1.0, 1.0]
        bottomright = [self.aspect_ratio, 1.0, 1.0]
        self.frustum = np.array(
            [
                origin,
                topleft,
                origin,
                topright,
                origin,
                bottomleft,
                origin,
                bottomright,
                topleft,
                topright,
                topright,
                bottomright,
                bottomright,
                bottomleft,
                bottomleft,
                topleft,
            ],
            dtype=np.float32,
        )

    def add(self, T_WC, thickness=3, scale=1, color=None):
        frustum = T_WC.act(torch.from_numpy(self.frustum * scale)).numpy()
        thickness = np.ones_like(frustum[..., :1]) * thickness
        frustum = np.concatenate([frustum, thickness], axis=-1).reshape(-1, 4)
        color = [1.0, 1.0, 1.0, 1.0] if color is None else color
        colors = np.tile(color, (frustum.shape[0], 1)).astype(np.float32)
        self.lines.append(frustum)
        self.colors.append(colors)

    def render(self, camera, mode=None):
        if len(self.lines) == 0:
            return
        self.lines = np.concatenate(self.lines, axis=0)
        self.colors = np.concatenate(self.colors, axis=0)
        self.clear()
        super().render(camera, mode=mode)
        self.lines = []
        self.colors = []


class Lines(LineGeometry):
    def __init__(self, program):
        super().__init__()
        self.program = program
        self.lines = []
        self.colors = []

    def add(self, start, end, thickness=1, color=None):
        start = start.reshape(-1, 3).astype(np.float32)
        end = end.reshape(-1, 3).astype(np.float32)

        thickness = np.ones_like(start[..., :1]) * thickness
        start_xyzw = np.concatenate([start, thickness], axis=-1)
        end_xyzw = np.concatenate([end, thickness], axis=-1)
        line = np.concatenate([start_xyzw, end_xyzw], axis=1).reshape(-1, 4)
        if isinstance(color, np.ndarray):  # TODO Bit hacky!
            colors = color.reshape(-1, 4).astype(np.float32)
        else:
            color = [1.0, 1.0, 1.0, 1.0] if color is None else color
            colors = np.tile(color, (line.shape[0], 1)).astype(np.float32)

        # make sure that the dimensions match!
        assert line.shape[0] == colors.shape[0]
        assert line.shape[1] == 4 and colors.shape[1] == 4

        self.lines.append(line)
        self.colors.append(colors)

    def render(self, camera, mode=None):
        if len(self.lines) == 0:
            return
        self.lines = np.concatenate(self.lines, axis=0)
        self.colors = np.concatenate(self.colors, axis=0)
        self.clear()
        super().render(camera, mode=mode)
        self.lines = []
        self.colors = []


def image_with_text(img, size, text, same_line=False):
    # check if the img is too small to render
    if size[0] < 16:
        return
    text_cursor_pos = imgui.get_cursor_pos()
    imgui.image(img.texture.glo, *size)
    if same_line:
        imgui.same_line()
    next_cursor_pos = imgui.get_cursor_pos()
    imgui.set_cursor_pos(text_cursor_pos)
    imgui.text(text)
    imgui.set_cursor_pos(next_cursor_pos)
