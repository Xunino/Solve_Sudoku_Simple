from model import sudokuNet
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import mnist
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report

INIT_LR = 1e-3
EPOCHS = 20
BS = 32

print("[INFO] Load data....")
((trainData, trainLabels), (testData, testLabels)) = mnist.load_data()
# print(trainData.shape)

# add channel dimension
trainData = trainData.reshape((trainData.shape[0], 28, 28, 1))
testData = testData.reshape((testData.shape[0], 28, 28, 1))

# Scale [0, 1]
trainData = trainData.astype("float32") / 255.0
testData = testData.astype("float32") / 255.0

# Chuyển đổi labels từ số sang vector
lb = LabelBinarizer()
trainLabels = lb.fit_transform(trainLabels)
testLabels = lb.transform(testLabels)
# print(trainLabels[0])

# Khởi tạo model
opt = Adam(learning_rate=INIT_LR)
model = sudokuNet.build(width=28, height=28, depth=1, classes=10)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["acc"])

print("[INFO] Training the model....")
H = model.fit(
    trainData,
    trainLabels,
    validation_data=[testData, testLabels],
    epochs=EPOCHS,
    batch_size=BS,
    verbose=1
)

# Evaluate the network
print("[INFO] Evaluate network....")
predicted = model.predict(testData)
print(classification_report(testLabels.argmax(axis=1),
                            predicted.argmax(axis=1),
                            target_names=[str(x) for x in lb.classes_]
                            ))
print("[INFO] Saving the model....")
model.save("digit_classifier.h5")
print("[INFO] DONE...!")