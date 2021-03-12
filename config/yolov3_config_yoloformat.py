EVAL_REUSLTS_DIR = f"C:/Users/theppitak.sarut/Desktop/repVGG_YOLO3/coco128/valid_results"
TEST_REUSLTS_DIR = f"C:/Users/theppitak.sarut/Desktop/repVGG_YOLO3/test_data/test_results"

# DATA = {"CLASSES":['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
#            'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
#            'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
#            'train', 'tvmonitor'],
#         "NUM":20}


DATA = {"CLASSES":['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
        'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
        'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
        'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
        'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
        'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
        'hair drier', 'toothbrush'],
        "NUM":80}

# model
MODEL = {"ANCHORS":[[(1.25, 1.625), (2.0, 3.75), (4.125, 2.875)],  # Anchors for small obj (1/8) >> most number of cells
            [(1.875, 3.8125), (3.875, 2.8125), (3.6875, 7.4375)],  # Anchors for medium obj
            [(3.625, 2.8125), (4.875, 6.1875), (11.65625, 10.1875)]] ,# Anchors for big obj  # all anchors is in the grid scale of it yolo layers
         "STRIDES":[8, 16, 32],
         "ANCHORS_PER_SCLAE":3
         }

# train
TRAIN = {
         "TRAIN_IMG_SIZE":288,
         "AUGMENT":True,
        #  "BATCH_SIZE":8,
         "BATCH_SIZE":4,
         "MULTI_SCALE_TRAIN":True, # If True then TRAIN_IMG_SIZE almost has no meaning it will use 256 only for 10 eporch
         "IOU_THRESHOLD_LOSS":0.5,
        #  "EPOCHS":300,
         "EPOCHS":100,
         "NUMBER_WORKERS":4,
         "MOMENTUM":0.9,
         "WEIGHT_DECAY":0.0005,
         "LR_INIT":1e-4,
         "LR_END":1e-6,
         "WARMUP_EPOCHS":0  # or 0  2
         } 

# eval
EVAL = {
        "TEST_IMG_SIZE":288,
        "BATCH_SIZE":1,   
        "NUMBER_WORKERS":0, 
        "CONF_THRESH":0.01, 
        "NMS_THRESH":0.5,     # IOU thrs during nms
        "MULTI_SCALE_TEST":False,
        "FLIP_TEST":False,
        "SHOW_RESULT": False,
        "SAVE_RESULT": True
        }


# test
TEST = {
        "TEST_IMG_SIZE":288,
        "BATCH_SIZE":1,   #Fix
        "NUMBER_WORKERS":0, # Fix
        "CONF_THRESH":0.01, 
        "NMS_THRESH":0.5,   # IOU thrs during nms
        "MULTI_SCALE_TEST":False, # Fix
        "FLIP_TEST":False,
        "SHOW_RESULT": False,
        "SAVE_RESULT": True
        }