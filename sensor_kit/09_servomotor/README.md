# 🎥 Face Tracking Pan-Tilt Camera System

OpenCV와 STM32F103을 활용한 실시간 얼굴 추적 Pan-Tilt 카메라 시스템

## 📋 개요

웹캠으로 얼굴을 감지하고, 서보모터를 제어하여 얼굴을 화면 중앙에 자동으로 유지하는 시스템입니다.

## ✨ 주요 기능

- **실시간 얼굴 추적**: Haar Cascade 알고리즘을 사용한 빠른 얼굴 감지
- **자동 Pan-Tilt 제어**: STM32F103과 시리얼 통신으로 서보모터 제어
- **데드존 설정**: 불필요한 미세 움직임 방지
- **다중 얼굴 처리**: 가장 큰 얼굴(가장 가까운 사람) 자동 선택
- **실시간 시각화**: 추적 상태, 오차, 데드존 표시

## 🛠️ 하드웨어 요구사항

- STM32F103 (Blue Pill 등)
- 2축 Pan-Tilt 서보 메커니즘 (SG90 등)
- 웹캠
- USB-to-Serial 어댑터 (또는 STM32 내장 USB)

## 💻 소프트웨어 요구사항

```bash
pip install opencv-python pyserial
```

**Python 버전**: 3.7 이상

## 🚀 사용법

### 1. 카메라 확인

<img width="1462" height="302" alt="캠테1" src="https://github.com/user-attachments/assets/dfde52b4-7578-47d0-8e97-7a787c67fe95" />

<img width="1155" height="626" alt="캠테2" src="https://github.com/user-attachments/assets/b8f73880-c5a6-424b-8ad9-f934858f79e3" />

```bash
python camtest.py
```

연결된 카메라 ID를 확인합니다.

### 2. 얼굴 추적 실행

<img width="802" height="637" alt="캠테3" src="https://github.com/user-attachments/assets/9dd478db-9ecc-44b4-a9dc-4f8fcfad4894" />

![캠테4](https://github.com/user-attachments/assets/93dbcb5d-54d1-4f45-a288-4b3c9bc9b135)


**Windows:**
```bash
python face_tracking_pantilt.py --port COM3
```

**Linux:**
```bash
python face_tracking_pantilt.py --port /dev/ttyUSB0
```

### 3. 옵션

```bash
python face_tracking_pantilt.py \
    --port COM3 \
    --camera 1 \              # 카메라 ID
    --baudrate 115200 \       # 시리얼 통신 속도
    --deadzone 50 \           # 데드존 크기 (픽셀)
    --interval 0.1            # 명령 전송 간격 (초)
```

## ⌨️ 키보드 컨트롤

| 키 | 기능 |
|---|---|
| `Q` | 프로그램 종료 |
| `C` | 카메라 중앙으로 리셋 |
| `+` | 데드존 증가 |
| `-` | 데드존 감소 |

## 🔧 STM32 펌웨어

STM32는 다음 시리얼 명령을 수신해야 합니다:

- `w`: Tilt 위로
- `s`: Tilt 아래로
- `a`: Pan 오른쪽
- `d`: Pan 왼쪽
- `i`: 초기 위치 (중앙)

**시리얼 설정**: 115200 baud, 8N1

## 📊 동작 원리

1. 웹캠에서 프레임 캡처
2. Haar Cascade로 얼굴 감지
3. 얼굴 중심과 화면 중심 간의 오차 계산
4. 데드존 내부면 유지, 외부면 서보 이동 명령 전송
5. 시리얼 통신으로 STM32에 명령 전달
6. 서보모터 구동으로 카메라 방향 조정

## 📁 파일 구조

```
├── face_tracking_pantilt.py  # 메인 추적 프로그램
├── camtest.py                # 카메라 테스트 유틸리티
└── README.md
```

## ⚙️ 주요 파라미터

```python
# TrackingConfig 클래스
baudrate = 115200          # 시리얼 통신 속도
camera_id = 0              # 웹캠 ID
frame_width = 640          # 프레임 너비
frame_height = 480         # 프레임 높이
deadzone_x = 50            # 수평 데드존 (픽셀)
deadzone_y = 40            # 수직 데드존 (픽셀)
command_interval = 0.1     # 명령 전송 간격 (초)
min_face_size = 80         # 최소 얼굴 크기 (픽셀)
```

## 🐛 문제 해결

**카메라가 열리지 않음:**
- `camtest.py`로 카메라 ID 확인
- `--camera` 옵션으로 올바른 ID 지정

**시리얼 포트 오류:**
- Windows: 장치 관리자에서 COM 포트 확인
- Linux: `ls /dev/ttyUSB*` 또는 `ls /dev/ttyACM*`

**얼굴 감지 안 됨:**
- 조명 확인 (밝은 환경 권장)
- `min_face_size` 값 조정
- 카메라와의 거리 조정

## 📝 라이선스

MIT License

## 🤝 기여

이슈 및 PR 환영합니다!

---

**제작**: STM32 + OpenCV Face Tracking Project








