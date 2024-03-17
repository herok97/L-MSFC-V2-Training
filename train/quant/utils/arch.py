model_arch = {
    "faster_rcnn_X_101_32x8d_FPN_3x": {
        "cfg": "models/detectron2/configs/COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml",
        "weight": "weights/detectron2/COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x/139173657/model_final_68b088.pkl",
        "splits": ["p2", "p3", "p4", "p5"],
    },
    "mask_rcnn_X_101_32x8d_FPN_3x": {
        "cfg": "models/detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml",
        "weight": "weights/detectron2/COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x/139653917/model_final_2d9806.pkl",
        "splits": ["p2", "p3", "p4", "p5"],
    },
    "jde_1088x608": {
        "cfg": "models/Towards-Realtime-MOT/cfg/yolov3_1088x608.cfg",
        "weight": "weights/jde/jde.1088x608.uncertainty.pt",
        "iou_thres": 0.5,
        "conf_thres": 0.5,
        "nms_thres": 0.4,
        "min_box_area": 200,
        "track_buffer": 30,
        "frame_rate": 30  # It is odd to consider this at here but following original code.
    },
}