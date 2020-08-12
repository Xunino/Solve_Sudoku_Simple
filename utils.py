from imutils.perspective import four_point_transform
from skimage.segmentation import clear_border
import cv2
import imutils
import numpy as np

def find_puzzle(image, debug=False):
    # Chuyển đổi ảnh sang ảnh đen trắng
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 3)

    # Tìm ngưỡng
    threshold = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    threshold = cv2.bitwise_not(threshold)

    # Kiểm tra ảnh đã xử lý hay chưa
    if debug:
        cv2.imshow("Solver threshold", threshold)
        cv2.waitKey(0)

    # Tìm đường bao trong ảnh và sắp xếp chúng giảm dần
    contours = cv2.findContours(threshold.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Nối đường bao lại với nhau
    contours = imutils.grab_contours(contours)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # Khởi tạo biến
    puzzle_contours = None
    for c in contours:
        # Ước lượng số đường bao
        length = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * length, True)
        # print(approx)
        # Ta đi tìm giá trị 4 điểm của hình chữ nhật
        if len(approx) == 4:
            puzzle_contours = approx
            break

    if puzzle_contours is None:
        raise Exception(("Không tìm thấy khung sudoku"
                        "Vui lòng kiểm tra lại bằng cách debug."))
    # Kiểm tra
    if debug:
        output = image.copy()
        cv2.drawContours(output, [puzzle_contours], -1, (255, 0, 255), 2)
        cv2.imshow("Puzzle", output)
        cv2.waitKey(0)

    # áp dụng phép biến đổi bối cảnh, đưa ảnh về ảnh vuông góc với mắt nhìn
    puzzle = four_point_transform(image, puzzle_contours.reshape(4, 2))
    warped = four_point_transform(gray, puzzle_contours.reshape(4, 2))

    if debug:
        cv2.imshow("Transform", puzzle)
        cv2.waitKey(0)
    return puzzle, warped


def extract_digit(cell, debug=False):

    thresh = cv2.threshold(cell, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    thresh = clear_border(thresh)

    if debug:
        cv2.imshow("Cell threshold", thresh)
        cv2.waitKey(0)

    # Tìm đường bao
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Nối đường bao lại với nhau
    cnts = imutils.grab_contours(cnts)
    # print(contours)
    if len(cnts) == 0:
        return None

    # Tìm contour lớn nhất trong mỗi cell và thêm lớp mask cho nó
    c = max(cnts, key=cv2.contourArea)
    mask = np.zeros(thresh.shape, dtype="uint8")
    cv2.drawContours(mask, [c], -1, 255, -1)

    (h, w) = thresh.shape
    # Kiểm tra độ trùng khớp
    overlap = cv2.countNonZero(mask) / float(w * h)
    # print(cv2.countNonZero(mask))
    if overlap < 0.03:
        return None

    digit = cv2.bitwise_and(thresh, thresh, mask=mask)
    # print(digit)
    if debug:
        cv2.imshow("Digit", digit)
        cv2.waitKey(0)
    return digit