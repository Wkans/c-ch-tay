import cv2
import handtracking as htm
import pyautogui
import numpy as np
import time

last_click_time = 0
click_delay = 0.3

# Biến cuộn trang cải tiến
scroll_active = False
scroll_start_y = 0
scroll_threshold = 10  # Ngưỡng nhạy cuộn
scroll_smoothing = 3  # Làm mượt chuyển động cuộn
scroll_history = []  # Lưu lịch sử vị trí để làm mượt

# Khởi tạo webcam
cap = cv2.VideoCapture(0)
detector = htm.handDetector(maxHands=1)
fingers_number = [4, 8, 12, 16, 20]

# Biến để làm mượt chuyển động chuột
smoothening = 5
prev_x, prev_y = 0, 0
curr_x, curr_y = 0, 0

wCam, hCam = 800, 500
wScreen, hScreen = pyautogui.size()
drag = False

while True:
    success, frame = cap.read()
    if not success:
        break
    # Lật ngược camera
    image = cv2.flip(frame, 1)

    # Xác định vị trí bàn tay trên webcam
    image = detector.findHands(image)
    lm_list = detector.findPosition(image, draw=False)

    # Kiểm tra sự tồn tại của bàn tay
    if len(lm_list) != 0:
        # Xác định tọa độ ngón trỏ
        x1, y1 = lm_list[8][1:]

        # Xác định trạng thái các ngón tay
        fingers = []
        # Xác định ngón cái
        if lm_list[fingers_number[0]][1] < lm_list[fingers_number[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        # Xác định 4 ngón còn lại
        for id_finger in range(1, 5):
            if lm_list[fingers_number[id_finger]][2] < lm_list[fingers_number[id_finger] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        cv2.rectangle(image, (100, 50), (wCam - 250, hCam - 190), (0, 0, 0), 2)

        # Chuyển đổi tọa độ ngón trỏ tương ứng với màn hình máy tính
        x3 = np.interp(x1, (100, wCam - 200), (0, wScreen))
        y3 = np.interp(y1, (50, hCam - 190), (0, hScreen))

        # Làm mịn các giá trị
        curr_x = prev_x + (x3 - prev_x) / smoothening
        curr_y = prev_y + (y3 - prev_y) / smoothening

        # Tắt di chuột
        if drag == True and fingers.count(0) != 5:
            drag = False
            pyautogui.mouseUp(button="left")

        # Chỉ ngón trỏ có chế độ di chuyển
        if fingers[0] == 0 and fingers[1] == 1 and fingers[2] == 0 and fingers[3] == 0 and fingers[4] == 0:
            pyautogui.moveTo(curr_x, curr_y)
            cv2.circle(image, (x1, y1), 10, (0, 255, 0), -1)
            prev_x, prev_y = curr_x, curr_y

        # Click chuột trái khi chỉ có ngón trỏ, ngón giữa được duỗi
        elif fingers[0] == 0 and fingers[1] == 1 and fingers[2] == 1 and fingers[3] == 0 and fingers[4] == 0:
            length, img, lineInfo = detector.findDistance(8, 12, image)
            if length < 27:
                if time.time() - last_click_time > click_delay:
                    cv2.circle(image, (lineInfo[4], lineInfo[5]), 10, (0, 255, 0), -1)
                    pyautogui.click(button="left")
                    last_click_time = time.time()

        # Click chuột phải khi ngón trỏ, ngón giữa, ngón cái được duỗi
        elif fingers[0] == 1 and fingers[1] == 1 and fingers[2] == 1 and fingers[3] == 0 and fingers[4] == 0:
            length, img, lineInfo = detector.findDistance(8, 12, image)
            if length < 27:
                cv2.circle(image, (lineInfo[4], lineInfo[5]), 10, (0, 255, 0), -1)
                pyautogui.click(button="right")


        # Kích hoạt cuộn khi tất cả ngón tay đều duỗi (bàn tay mở)
        if all(fingers):
            middle_finger_tip = lm_list[12][2]  # Y coordinate của đầu ngón giqqữa

            if not scroll_active:
                scroll_start_y = middle_finger_tip
                scroll_active = True
                scroll_history = [middle_finger_tip] * scroll_smoothing  # Khởi tạo lịch sử
            else:
                # Thêm vị trí mới vào lịch sử
                scroll_history.append(middle_finger_tip)
                scroll_history = scroll_history[-scroll_smoothing:]  # Giữ chỉ số lượng mẫu cần thiết

                # Tính giá trị trung bình làm mượt
                avg_y = sum(scroll_history) / len(scroll_history)
                dy = avg_y - scroll_start_y

                # Kiểm tra ngưỡng cuộn
                if abs(dy) > scroll_threshold:
                    scroll_amount = int(-dy * 4)  # Nhân hệ số để tăng tốc độ cuộn
                    pyautogui.scroll(scroll_amount)
                    scroll_start_y = avg_y  # Cập nhật vị trí mới

                    # Hiển thị hướng cuộn trên màn hình
                    if dy > 0:
                        cv2.putText(image, "SCROLL DOWN", (50, 100),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    else:
                        cv2.putText(image, "SCROLL UP", (50, 100),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            scroll_active = False

    # Hiển thị camera
    cv2.imshow("Virtual Mouse", image)
    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()