"""
config.py — 配置文件加载与管理模块

功能描述:
    提供 YAML 配置文件的加载、继承合并和全局配置管理功能。
    支持通过 "inherit" 字段实现配置继承（子配置覆盖父配置的同名项）。

使用方式:
    from mast3r_fusion.config import load_config, config
    load_config("config/my_config.yaml")
    # 之后可以通过 config["key"] 访问任意配置项

全局变量:
    config: dict  — 全局配置字典，加载后在所有模块中共享
"""

import yaml         # YAML 文件解析库
import pathlib      # 路径操作工具

# 全局配置字典，加载后可通过 config["key"] 在所有模块中访问
config = {}


def load_config(filepath):
    """
    加载 YAML 配置文件，并支持配置继承。
    
    如果配置文件中包含 "inherit" 字段，则先递归加载父配置，
    然后用子配置的内容覆盖父配置中的同名键。
    
    处理流程:
    1. 读取指定的 YAML 文件
    2. 检查是否有 "inherit" 字段
    3. 如果有，先递归加载父配置文件
    4. 用子配置的内容更新（覆盖）全局 config 字典
    
    参数:
        filepath: str — YAML 配置文件路径
        
    副作用:
        修改全局变量 config
    """
    global config
    with open(filepath, "r") as f:
        yml = yaml.load(f, Loader=yaml.SafeLoader)

    # Handle inherit
    # 处理配置继承: 如果存在 "inherit" 字段，先递归加载父配置
    if "inherit" in yml:
        # 父配置的路径相对于当前配置文件所在目录
        parent_path = pathlib.Path(filepath).parent / yml["inherit"]
        load_config(parent_path)
        yml.pop("inherit")  # 移除 inherit 字段本身

    # 用当前配置更新全局 config（后加载的覆盖先前的）
    config.update(yml)


def set_global_config(cfg):
    """
    直接设置全局配置字典。
    
    在多进程场景中，子进程无法直接访问主进程加载的 config 字典，
    因此需要通过此函数将配置显式传递给子进程。
    
    参数:
        cfg: dict — 要设置为全局配置的字典
        
    使用场景:
        - 可视化进程启动时，主进程将 config 传递给子进程
        - 回环检测/全局优化脚本中加载配置后设置全局变量
    """
    global config
    config = cfg
