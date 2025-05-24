# thêm các thư viện cần thiết
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import os

# khai báo các giá trị tham số Learning rate, epoch, batch size
INIT_LR = 1e-4
EPOCHS = 20
BS = 50

# khai báo đường dẫn tới thư mục chứa data và lable 
DIRECTORY = r"C:\Users\hoang\OneDrive\Desktop\Face-Mask-Detection\dataset"
CATEGORIES = ["with_mask", "without_mask"]

# đọc tất cả các ảnh từ thư mục chứa dữ liệu 
# và đưa vào danh sách dữ liệu và các nhãn tương ứng
print("------------Dang tai hinh anh...---------------")
# khởi tạo mảng data và lables
data = []
labels = []

for category in CATEGORIES:
    path = os.path.join(DIRECTORY, category)
    for img in os.listdir(path):
    	img_path = os.path.join(path, img)
    	image = load_img(img_path, target_size=(224, 224))
    	image = img_to_array(image)
    	image = preprocess_input(image)

    	data.append(image)
    	labels.append(category)

# đổi danh sách ảnh data và mảng nhãn labels thành numpy array 
# để tiện cho việc huấn luyện mô hình
data = np.array(data, dtype="float32")
labels = np.array(labels)

# mã hóa one-hot các nhãn của dữ liệu 
# và chia dữ liệu thành tập train và test
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

(trainX, testX, trainY, testY) = train_test_split(data, labels,
		 test_size=0.20, stratify=labels, random_state=42)

# tạo một bộ tạo ảnh huấn luyện với các trường hợp tăng cường dữ liệu 
# (thay đổi tính chất ảnh để tăng thêm độ chính xác train)
aug = ImageDataGenerator(
	rotation_range=20,
	zoom_range=0.16,
	width_shift_range=0.2,
	height_shift_range=0.2,
	shear_range=0.15,
	horizontal_flip=True,
	fill_mode="nearest")

# Tạo mô hình mạng neural network sử dụng kiến trúc MobileNetV2 
# và thêm một số lớp Fully Connected
baseModel = MobileNetV2(weights="imagenet", include_top=False, 
			input_tensor=Input(shape=(224, 224, 3)))

# xây dựng phần đầu của mô hình phân loại (custom)
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(128, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2, activation="softmax")(headModel)

# kết hợp phần đầu của mô hình phân loại (output) 
# với base model MobileNetV2 (input)
model = Model(inputs=baseModel.input, outputs=headModel)

# đóng băng các lớp trong base model, 
# để trọng số của mô hình MobileNet không bị cập nhật (tránh overfiting)
for layer in baseModel.layers:
	layer.trainable = False

# compile mô hình với hàm loss binary crossentropy và optimizer Adam
print("------------Dang tien hanh bien dich mo hinh...---------------")
opt = Adam(learning_rate=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt,
	metrics=["accuracy"])

# huấn luyện mô hình với các tham số tương ứng
print("------------Dang tien hanh qua trinh huan luyen...---------------")
H = model.fit(
	aug.flow(trainX, trainY, batch_size=BS),
	steps_per_epoch=len(trainX) // BS,
	validation_data=(testX, testY),
	validation_steps=len(testX) // BS,
	epochs=EPOCHS)

# dự đoán và đánh giá hiệu suất của mô hình trên tập test
print("------------Dang tien hanh danh gia hieu suat...---------------")
predIdxs = model.predict(testX, batch_size=BS)

# với mỗi ảnh trong tập test 
# thì cần tìm giá trị nhãn có độ chính xác cao nhất tương ứng
predIdxs = np.argmax(predIdxs, axis=1)

# tạo báo cáo phân loại
print(classification_report(testY.argmax(axis=1), 
			    predIdxs, target_names=lb.classes_))

# lưu trữ mô hình mạng đã được huấn luyện
print("------------Dang luu mo hinh...---------------")
model.save("mask_detect.model", save_format="h5")

# tạo đồ thị biểu diễn sự tiến triển của độ chính xác 
# và hàm loss trong quá trình huấn luyện
N = EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig("plot.png")