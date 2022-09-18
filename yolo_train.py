from yolov5 import train 

train.run(data='train.yaml', epochs=5, weights='yolov5s.pt')