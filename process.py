from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
import cv2
from sudoku import Sudoku
from utils import find_puzzle, extract_digit

# ---------------------------------------------------
def solve(image, model, debug=False, method="A", nextChoice="1"):
    # tìm kiếm các ô thiếu, sau đó
    (puzzleImage, warped) = find_puzzle(image, debug=debug)

    # Khởi tạo board sudoku
    size = (9, 9)
    board = np.zeros(size, dtype="int")

    # Đưa ảnh về 81 ô
    stepX = warped.shape[1] // size[1]
    stepY = warped.shape[0] // size[0]

    # Tìm các ô có giá trị trong bảng sudoku
    # Khởi tạo
    cellLocs = []
    for y in range(0, 9):
        # Khởi tạo biến lưu thông tin hàng
        row = []

        for x in range(0, 9):
            # tính toán toạ độ của ô chứa thông tin hiện tại
            startX = x * stepX
            startY = y * stepY
            endX = (x + 1) * stepX
            endY = (y + 1) * stepY

            # thêm toạ độ (x, y)
            row.append((startX, startY, endX, endY))

            # Cắt ảnh từ ảnh chuyển đổi sau đó trích xuất các số
            cell = warped[startY:endY, startX:endX]
            digit = extract_digit(cell, debug=debug)

            # Kiểm tra
            if digit is not None:
                # thay đổi kích thước ảnh
                roi = cv2.resize(digit, (28, 28))
                roi = roi.astype("float") / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)

                # Phân loại số
                pred = model.predict(roi).argmax(axis=1)[0]
                board[y, x] = pred

        # lưu trữ
        cellLocs.append(row)

    # ---------------------------------------------------
    # Khởi tạo model Sudoku
    from solve_sudoku import Sudoku
    import time

    print(board)
    puzzle = board.tolist()
    mySudoku = Sudoku(puzzle)


    # Giải quyết bài toán
    solution = mySudoku.solve(method, nextChoice, False)
    # print(solution)

    # Kết quả
    temp = np.array(solution)
    # print(temp)
    results = temp - board
    # lặp qua cell locationsvà board
    for (cellRow, boardRow) in zip(cellLocs, results):
        # print(cellRow, boardRow)
        # vùng tương ứng với số cần điền
        for (box, digit) in zip(cellRow, boardRow):
            startX, startY, endX, endY = box

            # Tính toán toạ độ
            textX = int((endX - startX) * 0.35)
            textY = int((endY - startY) * -0.28)
            textX += startX
            textY += endY

            # Vẽ kết quả
            if digit > 0:
                cv2.putText(puzzleImage, str(digit), (textX, textY), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 2)

    return puzzleImage