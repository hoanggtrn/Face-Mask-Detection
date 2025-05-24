# thêm các thư viện cần thiết
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import imutils
import time
import cv2
import os

# xác định vị trí và phát hiện khẩu trang trên các khuôn mặt trong khung hình
def detect_and_predict_mask(frame, faceNet, maskNet):
	# lấy kích thước khung hình sau đó xây dựng blob (input) trên đó
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224), (104.0, 177.0, 123.0))

	# truyền đầu vào (blob) through the network and obtain the face detections
	faceNet.setInput(blob)
	detections = faceNet.forward()
	print(detections.shape)

	# khởi tạo danh sách các khuôn mặt, 
	# vị trí tương ứng và danh sách các dự đoán từ mạng phân loại khẩu trang.
	faces = []
	locs = []
	preds = []

	# lặp qua các kết quả phát hiện khuôn mặt được lưu trữ trong numpy array detections
	for i in range(0, detections.shape[2]):
		# trích xuất độ tin cậy (confidence) tương ứng với kết quả phát hiện
		confidence = detections[0, 0, i, 2]

		# lọc các kết quả phát hiện khuôn mặt yếu bằng cách 
		# đảm bảo rằng độ tin cậy của chúng lớn hơn ngưỡng độ tin cậy tối thiểu
		if confidence > 0.5:
			# tính toán tọa độ của hộp giới hạn (bounding box) 
			# tương ứng với khuôn mặt được phát hiện
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# đảm bảo rằng các hộp giới hạn của khuôn mặt được phát hiện 
			# không vượt quá kích thước của khung hình
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

			# trích xuất các đặc điểm khuôn mặt được nhận dạng, 
			# chuyển đổi kênh màu từ BGR sang RGB 
			# và thay đổi kích thước của khuôn mặt thành (224, 224) 
			# sau đó đc tiền xử lý
			face = frame[startY:endY, startX:endX]
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = cv2.resize(face, (224, 224))
			face = img_to_array(face)
			face = preprocess_input(face)

			#  thêm khuôn mặt và hộp giới hạn vào các danh sách tương ứng
			faces.append(face)
			locs.append((startX, startY, endX, endY))

	# chỉ thực hiện dự đoán khi có ít nhất 1 khuôn mặt trong khung hình
	if len(faces) > 0:
		# dự đoán trên tất cả các khuôn mặt cùng một lúc để tăng tốc độ xử lý
		faces = np.array(faces, dtype="float32")
		preds = maskNet.predict(faces, batch_size=32)

	# trả về một tuple gồm hai phần tử là 
	# danh sách các hộp giới hạn cho mỗi khuôn mặt 
	# và danh sách các xác suất được dự đoán cho từng khuôn mặt tương ứng
	return (locs, preds)

# tải mô hình phát hiện khuôn mặt, khởi tạo một đối tượng faceNet
prototxtPath = r"face_detector\deploy.prototxt"
weightsPath = r"face_detector\res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# tải mô hình nhận diện đã được huấn luyện trước đó
maskNet = load_model("mask_detect.model")

# khởi tạo trình video
print("--------------Dang khoi tao luong video...----------------")
vs = VideoStream(src=0).start()

# lặp lại cho từng khung hình trên video
while True:
	# lấy từng khung hình từ luồng video và điều chỉnh kích thước đọc được thành 1024x768
	frame = vs.read()
	frame = imutils.resize(frame, width=800, height=600)

	# phát hiện khuôn mặt và dự đoán xem mỗi khuôn mặt có đang đeo khẩu trang hay không
	(locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

	# duyệt qua danh sách các hộp giới hạn cho từng khuôn mặt (locs) 
	# và các xác suất được dự đoán cho từng khuôn mặt tương ứng (preds)
	for (box, pred) in zip(locs, preds):
		# giải nén các tọa độ của hộp giới hạn 
		# và xác suất của một khuôn mặt có đeo khẩu trang hay không
		(startX, startY, endX, endY) = box
		(mask, withoutMask) = pred

		# xác định nhãn lớp và màu sắc sẽ được sử dụng 
		# để vẽ hộp giới hạn và nhãn dự đoán trên khuôn mặt
		label = "Phat hien khau trang" if mask > withoutMask else "Khong phat hien khau trang"
		color = (0, 255, 0) if label == "Phat hien khau trang" else (0, 0, 255)

		# cập nhật nhãn dự đoán để bao gồm xác suất của lớp được dự đoán
		label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

		# vẽ nhãn dự đoán lên khung hình và vẽ hộp giới hạn xung quanh khuôn mặt
		cv2.putText(frame, label, (startX, startY - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
		cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

	# hiển thị khung hình đầu ra
	cv2.imshow("Realtime_mask_detect", frame)
	key = cv2.waitKey(1) & 0xFF

	# chờ người dùng nhấn phím để thoát khỏi chương trình
	if key == ord("q"):
		break

# dọn dẹp tài nguyên sau khi chương trình kết thúc
cv2.destroyAllWindows()
vs.stop()