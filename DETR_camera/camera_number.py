print('接続されているカメラの番号を調べています...')
true_camera_is = []  # 空の配列を用意
cam_number = []

    # カメラ番号を0～9まで変えて、COM_PORTに認識されているカメラを探す
for camera_number in range(0, 10):
    try:
        cap = cv2.VideoCapture(camera_number)
        ret, frame = cap.read()
    except:
        ret = False
    if ret == True:
        true_camera_is.append(camera_number)
        print("カメラ番号->", camera_number, "接続済")

        cam_number.append(camera_number)
    else:
        print("カメラ番号->", camera_number, "未接続")
print("接続されているカメラは", len(true_camera_is), "台です。")
print("カメラのインデックスは", true_camera_is,"です。")

