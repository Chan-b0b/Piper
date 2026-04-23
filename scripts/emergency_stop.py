#!/usr/bin/env python3
# -*-coding:utf8-*-

"""
급하게 PiPER를 멈추고 싶을 경우, 해당 파일을 꼭 실행해야 합니다!
"""

from typing import Optional
from piper_sdk import *

if __name__ == "__main__":
    piper = C_PiperInterface("can2")
    piper.ConnectPort()
    
    piper.MotionCtrl_1(0x01,0,0x00) # 긴급 정지버튼 (E-stop)
    pass