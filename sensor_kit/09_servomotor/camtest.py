
"""
카메라 확인 프로그램
- 연결된 카메라 자동 검색
- 다중 카메라 동시 표시
"""

import cv2

def find_cameras(max_check=5):
    """연결된 카메라 검색"""
    cameras = []
    for i in range(max_check):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                cameras.append(i)
                print(f"[OK] 카메라 {i} 발견")
            cap.release()
    return cameras

def main():
    print("카메라 검색 중...")
    cameras = find_cameras()
   
    if not cameras:
        print("카메라를 찾을 수 없습니다!")
        return
   
    print(f"\n총 {len(cameras)}개 카메라 발견: {cameras}")
    print("ESC 또는 Q: 종료\n")
   
    # 카메라 열기
    caps = {}
    for cam_id in cameras:
        cap = cv2.VideoCapture(cam_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        caps[cam_id] = cap
   
    while True:
        for cam_id, cap in caps.items():
            ret, frame = cap.read()
            if ret:
                # 카메라 번호 표시
                cv2.putText(frame, f"Camera {cam_id}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow(f"Camera {cam_id}", frame)
       
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):  # ESC or Q
            break
   
    # 정리
    for cap in caps.values():
        cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()