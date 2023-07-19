""" ━━━━━━神兽出没━━━━━━ 
 　　　┏┓　　　┏┓ 
 　　┃　　　　　　　┃ 
 　　┃　　　━　　　┃ 
 　　┃　┳┛　┗┳　┃ 
 　　┃　　　　　　　┃ 
 　　┃　　　┻　　　┃ 
 　　┃　　　　　　　┃ 
 　　┗━┓　　　┏━┛Code is far away from bug with the animal rotecting 
 　　　　┃　　　┃ 神兽保佑,代码无bug 
 　　　　┃　　　┃ 
 　　　　┃　　　┗━━━┓ 
 　　　　┃　　　　　　　┣┓ 
 　　　　┃　　　　　　　┏┛ 
 　　  　┗┓┓┏━┳┓┏┛ 
　　　　　┃┫┫　┃┫┫ 
　　　　　┗┻┛　┗┻┛ 
"""
import argparse
import os
import platform
import sys
from pathlib import Path

import torch

FILE = Path(__file__).resolve()# __file__指的是当前文件(即detect.py),FILE最终保存着当前文件的绝对路径,比如D://yolov5/detect.py
ROOT = FILE.parents[0]   #  ROOT保存着当前项目的父目录,比如 D://yolov5
print(ROOT)
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
print(sys.path)
ROOT = Path(os.path.relpath(ROOT, Path.cwd())) # ROOT设置为相对路径
print(ROOT)
print(__file__)
