import math
import torch, torchvision
from torch import nn
import torchvision.transforms as T
from torchvision.models import resnet50
import requests
import matplotlib.pyplot as plt
from PIL import Image
import ipywidgets as widgets
import base64
import io
import IPython
from io import BytesIO

import cv2   #追加
import time  #追加

import numpy as np

print(torch.__version__)         # 1.9.0+cu102
print(torchvision.__version__)   # 0.10.0+cu102

import pyrealsense2 as rs
from realsensecv import RealsenseCapture

CLASSES = [
    'N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
    'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack',
    'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass',
    'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
    'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A',
    'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]


COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]



transform = T.Compose([
    T.Resize(800),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def box_cxcywh_to_xyxy(x):
    """
    (center_x, center_y, width, height)から(xmin, ymin, xmax, ymax)に座標変換
    """
    # unbind(1)でTensor次元を削除
    # (center_x, center_y, width, height)*N → (center_x*N, center_y*N, width*N, height*N)
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    # (center_x, center_y, width, height)*N の形に戻す
    return torch.stack(b, dim=1)

def rescale_bboxes(out_bbox, size):
    """
    バウンディングボックスのリスケール
    """
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    # バウンディングボックスの[0～1]から元画像の大きさにリスケール
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b
    
def put_rect(cv2_img, prob, boxes):
    colors = COLORS * 100
    output_image = cv2_img
    for p, (xmin, ymin, xmax, ymax), c in zip(prob, boxes.tolist(), colors):
        xmin = (int)(xmin)
        ymin = (int)(ymin)
        xmax = (int)(xmax)
        ymax = (int)(ymax)
        #print(xmin,ymin,xmax,ymax) 数値入力はint型のみ、除算後intへ変換する必要あり
        #中央の位置情報取得用のxcとyc
        xc=(xmin+xmax)/2
        yc=(ymin+ymax)/2
        xc=(int)(xc)
        yc=(int)(yc)
        #画像内の検出物体へのバウンディングボックス表示処理開始
        c[0],c[2]=c[2],c[0]
        c = tuple([(int)(n*255) for n in c])
        #バウンディングボックス作成
        output_image = cv2.rectangle(output_image,(xmin,ymin),(xmax,ymax),(0,0,255), 4)
        cl = p.argmax()
        #検出物体のクラスの名前をtextに格納
        text = f'{CLASSES[cl]}: {p[cl]:0.2f}'
        #名前用のバウンディングボックス
        output_image = cv2.rectangle(output_image,(xmin,ymin-20),(xmin+len(text)*10,ymin),(0,255,255),-1)
        #textを名前用バウンディングボックスの中座標へ入れる
        output_image = cv2.putText(output_image,text,(xmin,ymin-5),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),2)
        #中央（xc,yc）にマーカーを表示
        output_image = cv2.drawMarker(output_image,(xc,yc),(0, 255, 255),markerType=cv2.MARKER_CROSS,markerSize=15)
        #処理した画像を返す
    return output_image

align = rs.align(rs.stream.color)
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

#model load
model = torch.hub.load('facebookresearch/detr', 'detr_resnet50',pretrained=True)

#model.eval()
model = model.cuda()   #GPUを使用する場合はこちら

# プロパティの設定
seconds = 0.0
fps = 0.0

frame_rate = 30


width = 640
height = 480
size = (width, height)
THRESHOLD=1.0

pipeline = rs.pipeline()
profile=pipeline.start(config)

while True:
    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)
    color_frame = aligned_frames.get_color_frame()
    depth_frame = aligned_frames.get_depth_frame()
    if not depth_frame or not color_frame:
        continue

     
    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
    max_dist = THRESHOLD/depth_scale

    color_image = np.asanyarray(color_frame.get_data())
    depth_image = np.asanyarray(depth_frame.get_data())
    depth_filtered_image = (depth_image < max_dist) * depth_image
    color_filtered_image = (depth_filtered_image.reshape((height, width, 1)) > 0) * color_image

    # ヒートマップに変換

    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.08), cv2.COLORMAP_JET)


    # レンダリング
    #images = np.hstack((color_frame, color_filtered_image))  # RGBとDepthを横に並べて表示
    #cv2.imshow('RealSense', images)

    start = time.time() #FPS(Frame per Sec)計測用の時間
    #OpenCVは色の順番がBGR、PillowsはRGBを前提としているため変換が必要
    im =Image.fromarray(color_filtered_image)
    

    # mean-std normalize the input image (batch-size: 1)
    img = transform(im).unsqueeze(0).cuda()  #GPUを使用する場合
    #img = transform(im).unsqueeze(0)     #CPUしかなければこちら

    # propagate through the model
    with torch.no_grad(): #重みの更新があり得た際にその勾配の計算やグラフ化を行わないためのオプション
        outputs = model(img)  #画像をViTに入力して出力

    # keep only predictions with 0.7+ confidence
    probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]  #ここで0.9とかの確率を設定する　softmaxかけた結果で検出結果の確率を抽出
    keep = probas.max(-1).values > 0.9  #確率0.9以下をカット

    # convert boxes from [0; 1] to image scales
    bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, keep].cpu(), im.size) #GPU 出てきたボックスのリスケール
    #bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, keep], im.size)         #CPU

    #display
    output_image = put_rect(color_image, probas[keep], bboxes_scaled) #検出に使用した画像にリスケールした検出ボックスとその確率の記載

    # End time 一応終了
    end = time.time()   
    # Time elapsed
    seconds = (end - start) #1枚当たりの時間計算
    #print("time:{:.3f} msec".format(seconds*1000) )

    # Calculate frames per second
    fps  = ( fps + (1/seconds) ) / 2  #移動平均でFPSを導出(直前画像のFPSと処理した時間の平均)
    cv2.putText(output_image[0],'{:.2f}'.format(fps)+' fps',(10,50),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,0,0),3)  #画面左上にFPS記載
    images = np.hstack((output_image,depth_colormap ))  # RGBとDepthを横に並べて表示
    #cv2.imshow('RealSense', images)
    cv2.imshow(" ",output_image) #画像の出力コマンド
    #output_image[1]にxc,output_image[2]にycを格納
    #print("xc:",output_image[1],"yc=",output_image[2])
    
    # Press Q to stop!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ストリーミング停止
cv2.destroyAllWindows()

