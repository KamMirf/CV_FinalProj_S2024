# CV_FinalProj_S2024
Food detection and classification to recipe generation

Make sure you are in yolov5 directory:

python detect.py --weights ../weights/best.pt --source ../data/test/images --conf 0.25 --iou-thres 0.45 --data ../data/data.yaml --project ../yolo_outputs --name test_run --save-txt --save-conf > ../yolo_outputs/test_run/output.txt

the labels txt file stores data as:
```
<class> <x_center> <y_center> <width> <height> <confidence>
```