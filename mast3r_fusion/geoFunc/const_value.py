import math
"""
geoFunc/const_value.py — WGS-84 椭球体参数常量

定义 WGS-84 地球参考椭球的基本参数:
- a: 长半轴（赤道半径）= 6378137.0 米
- finv: 扁率的倒数 = 298.257223563
"""

pi=math.pi

# WGS-84 长半轴（赤道半径），单位: 米
a = 6378137.0
finv = 298.257223563