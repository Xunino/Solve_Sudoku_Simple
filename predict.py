from tensorflow.keras.models import load_model
from process import solve
import imutils
import cv2

# Tải dữ liệu
print("[INFO] loading digit classifier...")
model = load_model("digit_classifier.h5")

# Tải dữ liệu
print("[INFO] processing image...")
image = cv2.imread("images/9.jpg")
image = imutils.resize(image, width=600)

# result = solve(image, model, debug=False, method="A", nextChoice="9")
result = solve(image, model, debug=False)
cv2.imshow("Result", result)
cv2.waitKey(0)
cv2.imwrite("Output.jpg", result)

