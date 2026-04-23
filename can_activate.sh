#!/bin/bash

# 기본 CAN 이름으로 명령줄 인자를 통해 설정할 수 있습니다
DEFAULT_CAN_NAME="${1:-can0}"

# 단일 CAN 모듈의 기본 비트레이트이며, 명령줄 인자를 통해 설정할 수 있습니다.
DEFAULT_BITRATE="${2:-1000000}"

# USB 주소(옵션)
USB_ADDRESS="${3}"
echo "-------------------START-----------------------"
# ethtool 패키지 설치 여부 확인하기
if ! dpkg -l | grep -q "ethtool"; then
    echo "\e[31m오류: 시스템에서 ethtool 패키지 감지되지 않습니다.\e[0m"
    echo "아래 명령을 이용하여 ethtool을 설치해주세요:"
    echo "sudo apt update && sudo apt install ethtool"
    exit 1
fi

# can-utils 패키지 설치 여부 확인하기
if ! dpkg -l | grep -q "can-utils"; then
    echo "\e[31m오류: 시스템에서 can-utils 패키지 감지되지 않습니다.\e[0m"
    echo "아래 명령을 이용하여 can-utils을 설치해주세요:"
    echo "sudo apt update && sudo apt install can-utils"
    exit 1
fi

echo "ethtool와 can-utils 모두 설치 완료되어 있습니다."

# 현재 시스템에서 CAN 모듈 수 확인 후 변수 지정하기
CURRENT_CAN_COUNT=$(ip link show type can | grep -c "link/can")

if [ "$CURRENT_CAN_COUNT" -ne "1" ]; then  # 현재 시스템에서 CAN 모듈 수가 1인지 확인
    if [ -z "$USB_ADDRESS" ]; then  # 만약 USB 주소가 설정되지 않으면
        # 모든 CAN 인터페이스 확인
        for iface in $(ip -br link show type can | awk '{print $1}'); do  # CAN 인터페이스(can0, can1...)을 ifce 변수에 차례대로 넣기
            BUS_INFO=$(sudo ethtool -i "$iface" | grep "bus-info" | awk '{print $2}')  # 패키지 ethtool을 사용하여 bus-info 가져오기
            
            if [ -z "$BUS_INFO" ];then  # bus-info가 없을 경우
                echo "오류: $iface의 bus-info를 가져올 수 없습니다."
                continue
            fi
            
            echo "인터페이스 $iface가 USB 포트 $BUS_INFO에 삽입되어 있습니다."
        done
        echo -e " \e[31m 오류: 시스템에서 감지된 CAN 모듈의 수 ($CURRENT_CAN_COUNT)가 CAN 모듈 예상 수와 (1) 일치하지 않음\e[0m"
        echo -e " \e[31m USB 주소 매개변수를 추가하십시오. 예를 들어: \e[0m"
        echo -e " bash can_activate.sh can0 1000000 1-2:1.0"
        echo "-------------------ERROR-----------------------"
        exit 1
    fi
fi

# gs_usb 모듈 불러오기
# sudo modprobe gs_usb
# if [ $? -ne 0 ]; then
#     echo "오류: gs_usb 모듈을 불러올 수 없습니다."
#     exit 1
# fi

if [ -n "$USB_ADDRESS" ]; then
    echo "USB 주소 인자가 감지되었습니다: $USB_ADDRESS"
    
    # ethtool을 사용하여 USB 주소에 해당하는 CAN 인터페이스 찾기
    INTERFACE_NAME=""
    for iface in $(ip -br link show type can | awk '{print $1}'); do
        BUS_INFO=$(sudo ethtool -i "$iface" | grep "bus-info" | awk '{print $2}')
        if [ "$BUS_INFO" == "$USB_ADDRESS" ]; then
            INTERFACE_NAME="$iface"
            break
        fi
    done
    
    if [ -z "$INTERFACE_NAME" ]; then
        echo "오류: USB 주소 $USB_ADDRESS에 해당하는 CAN 인터페이스를 찾을 수 없습니다."
        exit 1
    else
        echo "USB 주소 $USB_ADDRESS에 해당하는 인터페이스 찾기: $INTERFACE_NAME"
    fi
else
    # 고유한 CAN 인터페이스 가져오기
    INTERFACE_NAME=$(ip -br link show type can | awk '{print $1}')
    
    # 인터페이스 이름을 찾았는지 검사하기
    if [ -z "$INTERFACE_NAME" ]; then
        echo "오류: CAN 인터페이스로 검출되지 않습니다."
        exit 1
    fi
    BUS_INFO=$(sudo ethtool -i "$INTERFACE_NAME" | grep "bus-info" | awk '{print $2}')
    echo "단일 can 모듈을 구성하여 인터페이스로 $INTERFACE_NAME 검출하기，상응하는 usb 주소는 $BUS_INFO"
fi

# 현재 인터페이스 활성화 여부 확인하기
IS_LINK_UP=$(ip link show "$INTERFACE_NAME" | grep -q "UP" && echo "yes" || echo "no")

# 현재 인터페이스의 비트레이트 가져오기
CURRENT_BITRATE=$(ip -details link show "$INTERFACE_NAME" | grep -oP 'bitrate \K\d+')

if [ "$IS_LINK_UP" == "yes" ] && [ "$CURRENT_BITRATE" -eq "$DEFAULT_BITRATE" ]; then
    echo "인터페이스 $INTERFACE_NAME이 활성화 되었으며, 비트레이트 $DEFAULT_BITRATE "
    
    # 인터페이스 이름이 기본 이름과 일치하는지 확인하기. 만약 인터페이스 이름과 CAN 이름이 다르면
    if [ "$INTERFACE_NAME" != "$DEFAULT_CAN_NAME" ]; then
        echo "인터페이스 $INTERFACE_NAME 이름을 $DEFAULT_CAN_NAME으로 변경합니다."
        sudo ip link set "$INTERFACE_NAME" down
        sudo ip link set "$INTERFACE_NAME" name "$DEFAULT_CAN_NAME"
        sudo ip link set "$DEFAULT_CAN_NAME" up
        echo "인터페이스는 $DEFAULT_CAN_NAME으로 이름이 변경되고 다시 활성화 되었습니다."
    else
        echo "인터페이스 이름은  이미 $DEFAULT_CAN_NAME로 설정되어 있습니다."
    fi
else
    # 인터페이스가 활성화되지 않았거나 비트레이트가 다를 경우
    if [ "$IS_LINK_UP" == "yes" ]; then
        echo "인터페이스 $INTERFACE_NAME이 활성화되었지만，비트레이트는 $CURRENT_BITRATE로 설정된 $DEFAULT_BITRATE와 일치하지 않습니다."
    else
        echo "인터페이스 $INTERFACE_NAME가 활성화되지 않았거나 비트레이트가 설정되지 않았습니다."
    fi
    
    # 인터페이스 비트레이트를 설정하고 활성화 시킨다.
    sudo ip link set "$INTERFACE_NAME" down
    sudo ip link set "$INTERFACE_NAME" type can bitrate $DEFAULT_BITRATE
    sudo ip link set "$INTERFACE_NAME" up
    echo "인터페이스 $INTERFACE_NAME가 비트레이트 $DEFAULT_BITRATE로 재설정되고 활성화 되었습니다."
fi

echo "-------------------OVER------------------------"
