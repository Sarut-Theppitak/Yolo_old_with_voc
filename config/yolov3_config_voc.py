
DATA_PATH = f"C:/Users/theppitak.sarut/Desktop/repVGG_YOLO3/vocdatasets/VOCdevkit"
PROJECT_PATH = f"C:/Users/theppitak.sarut/Desktop/repVGG_YOLO3"


DATA = {"CLASSES":['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
           'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
           'train', 'tvmonitor'],
        "NUM":20}

# model
MODEL = {"ANCHORS":[[(1.25, 1.625), (2.0, 3.75), (4.125, 2.875)],  # Anchors for small obj (1/8)
            [(1.875, 3.8125), (3.875, 2.8125), (3.6875, 7.4375)],  # Anchors for medium obj
            [(3.625, 2.8125), (4.875, 6.1875), (11.65625, 10.1875)]] ,# Anchors for big obj  # all anchors is in the grid scale of it yolo layers
         "STRIDES":[8, 16, 32],
         "ANCHORS_PER_SCLAE":3
         }

# train
TRAIN = {
         "TRAIN_IMG_SIZE":288,     # 448
         "AUGMENT":True,
         #"BATCH_SIZE":8,
         "BATCH_SIZE":4,
         "MULTI_SCALE_TRAIN":True,
         "IOU_THRESHOLD_LOSS":0.5,
         #"EPOCHS":50,
         "EPOCHS":300,
         "NUMBER_WORKERS":4,
         "MOMENTUM":0.9,
         "WEIGHT_DECAY":0.0005,
         "LR_INIT":1e-4,
         "LR_END":1e-6,
         "WARMUP_EPOCHS":2  # or None
         }


# test
TEST = {
        "TEST_IMG_SIZE":288,      #544
        "BATCH_SIZE":1,
        "NUMBER_WORKERS":0,
        "CONF_THRESH":0.01,
        "NMS_THRESH":0.5,
        "MULTI_SCALE_TEST":False,
        "FLIP_TEST":False
        }
