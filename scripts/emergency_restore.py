#!/usr/bin/env python3
# -*-coding:utf8-*-

"""
해당 파일은 0-2. piper_emergency_stop.py 파일을 실행시켰을 경우에만 적용됩니다.
해당 파일은 PiPER 모드를 리셋하기 위해 자동으로 Disable을 적용시키기 때문에 PiPER를 안전한 위치에 놓고 실행시켜주세요.
티칭 모드 적용 시, 현재는 리셋이 불가능합니다. 전원선을 뽑았다가 다시 재인가해주세요.
"""

from typing import Optional
import time, math
from piper_sdk import *

if __name__ == "__main__":
    piper = C_PiperInterface_V2("can2")
    piper.ConnectPort()

    piper.MotionCtrl_1(0x02, 0, 0x00)
    piper.MotionCtrl_1(0x00, 0, 0x00)

    piper.MotionCtrl_2(0x01, 0, 0, 0x00)  # 한 번만 실행 시 StandBy 모드
    piper.GripperCtrl(0, 0, 0x02, 0)
    time.sleep(1)

    piper.MotionCtrl_2(0x01, 0, 0, 0x00)  # 최종적으로 한번 더 실행해야 CAN 모드로 변경
    piper.GripperCtrl(0, 0, 0x03, 0)
    time.sleep(1)

    if piper.GetArmStatus().arm_status.ctrl_mode == 0x01:
        print("정상 리셋되었습니다. PiPER를 작동시킬 수 있습니다.")
    else:
        print("리셋에 실패했습니다. PiPER를 작동시키지 말아주세요!")
        pass
