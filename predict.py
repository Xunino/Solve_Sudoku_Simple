from tensorflow.keras.models import load_model
from process import solve
import imutils
import cv2

# initialize optimization functions based on the optimization parameters provided
"""
A - the first cell from left to right, from top to bottom
B - the first cell from right to left, from bottom to top
C - a randomly chosen cell
D - the closest cell to the center of the grid
E - the cell that currently has the fewest choices available
F - the cell that currently has the most choices available
G - the cell that has the fewest blank related cells
H - the cell that has the most blank related cells
I - the cell that is closest to all filled cells
J - the cell that is furthest from all filled cells
K - the cell whose related blank cells have the fewest available choices
L - the cell whose related blank cells have the most available choices
"""

# initialize optimization functions based on the optimization parameters provided
"""
0 - the lowest digit
1 - the highest digit
2 - a randomly chosen digit
3 - heuristically, the least used digit across the board
4 - heuristically, the most used digit across the board
5 - the digit that will cause related blank cells to have the least number of choices available
6 - the digit that will cause related blank cells to have the most number of choices available
7 - the digit that is the least common available choice among related blank cells
8 - the digit that is the most common available choice among related blank cells
9 - the digit that is the least common available choice across the board
a - the digit that is the most common available choice across the board
"""

# Tải dữ liệu
print("[INFO] loading digit classifier...")
model = load_model("digit_classifier.h5")

# Tải dữ liệu
print("[INFO] processing image...")
image = cv2.imread("images/8.JPG")
image = imutils.resize(image, width=600)

result = solve(image, model, debug=False, method="A", nextChoice="9")
cv2.imshow("Result", result)
cv2.waitKey(0)
cv2.imwrite("Output.jpg", result)

