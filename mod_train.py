import logging
import utils.gpu as gpu
from model.mod_yolov3 import Yolov3
from model.loss.yolo_loss import YoloV3Loss
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
from utils.mod_datasets import VocDataset, YoloDataset
import time
import shutil
import random
import argparse
from eval.mod_evaluator import VocEvaluator, YoloEvaluator
from utils.tools import *
from torch.utils.tensorboard import SummaryWriter
from utils import cosine_lr_scheduler
from utils.visualize import *


# import os
# os.environ["CUDA_VISIBLE_DEVICES"]='2'


class Trainer(object):
    def __init__(self,  cfg, train_dir, valid_dir, weight_path, resume, save_weight_dir, gpu_id):
        init_seeds(0)
        self.cfg = cfg
        self.train_dir = train_dir
        self.device = gpu.select_device(gpu_id)
        self.start_epoch = 0
        self.best_mAP = 0.
        self.epochs = cfg.TRAIN["EPOCHS"]
        self.weight_path = weight_path
        self.save_weight_dir = save_weight_dir
        self.multi_scale_train = cfg.TRAIN["MULTI_SCALE_TRAIN"]
        # self.train_dataset = VocDataset(cfg=cfg, anno_file_type="train", img_size=cfg.TRAIN["TRAIN_IMG_SIZE"])
        self.train_dataset = YoloDataset(cfg=cfg, train_dir=self.train_dir, img_size=cfg.TRAIN["TRAIN_IMG_SIZE"])
        self.train_dataloader = DataLoader(self.train_dataset,
                                           batch_size=cfg.TRAIN["BATCH_SIZE"],
                                           num_workers=cfg.TRAIN["NUMBER_WORKERS"],
                                           shuffle=True)
        self.yolov3 = Yolov3(cfg=cfg).to(self.device)
        # self.yolov3.apply(tools.weights_init_normal)

        self.optimizer = optim.SGD(self.yolov3.parameters(), lr=cfg.TRAIN["LR_INIT"],
                                   momentum=cfg.TRAIN["MOMENTUM"], weight_decay=cfg.TRAIN["WEIGHT_DECAY"])
        #self.optimizer = optim.Adam(self.yolov3.parameters(), lr = lr_init, weight_decay=0.9995)

        self.criterion = YoloV3Loss(anchors=cfg.MODEL["ANCHORS"], strides=cfg.MODEL["STRIDES"],
                                    iou_threshold_loss=cfg.TRAIN["IOU_THRESHOLD_LOSS"])

        self.__load_model_weights(weight_path, resume)

        self.scheduler = cosine_lr_scheduler.CosineDecayLR(self.optimizer,
                                                          T_max=self.epochs*len(self.train_dataloader),
                                                          lr_init=cfg.TRAIN["LR_INIT"],
                                                          lr_min=cfg.TRAIN["LR_END"],
                                                          warmup=cfg.TRAIN["WARMUP_EPOCHS"]*len(self.train_dataloader))

        self.valid_dir = valid_dir
        self.img_valid_dir = os.path.join(self.valid_dir, 'images')
        self.anno_valid_dir = os.path.join(self.valid_dir, 'labels')
        self.evaluator = YoloEvaluator(self.yolov3, self.cfg, self.img_valid_dir, "eval", anno_dir=self.anno_valid_dir)


    def __load_model_weights(self, weight_path, resume):
        if resume:
            last_weight = self.weight_path
            chkpt = torch.load(last_weight, map_location=self.device)
            self.yolov3.load_state_dict(chkpt['model'])
            self.start_epoch = chkpt['epoch'] + 1
            if chkpt['optimizer'] is not None:
                self.optimizer.load_state_dict(chkpt['optimizer'])
                self.best_mAP = chkpt['best_mAP']
            del chkpt
            print("loading weight, optimizer, epoch, and mAP from : {}".format(weight_path))
        else:
            if self.weight_path:  #train on new task but use someone weight. We will delete layers weights when NC is not the same, leading to the weight mismatch
                chkpt = torch.load(self.weight_path, map_location=self.device)
                del chkpt['_Yolov3__fpn._FPN_YOLOV3__conv0_1._Convolutional__conv.weight']
                del chkpt['_Yolov3__fpn._FPN_YOLOV3__conv0_1._Convolutional__conv.bias']
                del chkpt['_Yolov3__fpn._FPN_YOLOV3__conv1_1._Convolutional__conv.weight']
                del chkpt['_Yolov3__fpn._FPN_YOLOV3__conv1_1._Convolutional__conv.bias']
                del chkpt['_Yolov3__fpn._FPN_YOLOV3__conv2_1._Convolutional__conv.weight']
                del chkpt['_Yolov3__fpn._FPN_YOLOV3__conv2_1._Convolutional__conv.bias']
                self.yolov3.load_state_dict(chkpt, strict=False)
                del chkpt
                print("loading weight file from : {}".format(weight_path))
            
            else:
                print("training from scartch")
                # self.yolov3.load_darknet_weights(weight_path)


    def __save_model_weights(self, epoch, mAP):
        if mAP > self.best_mAP:
            self.best_mAP = mAP
        best_weight = os.path.join(self.save_weight_dir, "best.pt")
        last_weight = os.path.join(self.save_weight_dir, "last.pt")
        chkpt = {'epoch': epoch,
                 'best_mAP': self.best_mAP,
                 'model': self.yolov3.state_dict(),
                 'optimizer': self.optimizer.state_dict()}
        torch.save(chkpt, last_weight)

        if self.best_mAP == mAP:
            torch.save(chkpt['model'], best_weight)

        if epoch > 0 and epoch % 20 == 0:
            torch.save(chkpt, os.path.join(self.save_weight_dir, 'backup_epoch%g.pt'%epoch))
        del chkpt


    def train(self):
        print(self.yolov3)
        print("Train datasets number is : {}".format(len(self.train_dataset)))

        for epoch in range(self.start_epoch, self.epochs + 1):
            self.yolov3.train()

            mloss = torch.zeros(4)
            for i, (imgs, label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes)  in enumerate(self.train_dataloader):

                self.scheduler.step(len(self.train_dataloader)*epoch + i)

                imgs = imgs.to(self.device)
                label_sbbox = label_sbbox.to(self.device)
                label_mbbox = label_mbbox.to(self.device)
                label_lbbox = label_lbbox.to(self.device)
                sbboxes = sbboxes.to(self.device)
                mbboxes = mbboxes.to(self.device)
                lbboxes = lbboxes.to(self.device)

                p, p_d = self.yolov3(imgs)

                loss, loss_giou, loss_conf, loss_cls = self.criterion(p, p_d, label_sbbox, label_mbbox,
                                                  label_lbbox, sbboxes, mbboxes, lbboxes)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # Update running mean of tracked metrics
                loss_items = torch.tensor([loss_giou, loss_conf, loss_cls, loss])
                mloss = (mloss * i + loss_items) / (i + 1)  # mean

                # Print batch results
                if i % self.epochs == 0:
                    s = ('Epoch:[ %3d | %3d ]    Batch:[ %4d | %4d ]    loss_giou: %.4f    loss_conf: %.4f    loss_cls: %.4f    loss: %.4f    '
                         'lr: %g') % (epoch, self.epochs - 1, i, len(self.train_dataloader) - 1, mloss[0],mloss[1], mloss[2], mloss[3],
                                      self.optimizer.param_groups[0]['lr'])
                    print(s)

                # multi-sclae training (320-608 pixels) every 10 batches
                if self.multi_scale_train and (i+1)%10 == 0:
                    self.train_dataset.img_size = random.choice(range(10,20)) * 32
                    ##print("multi_scale_img_size : {}".format(self.train_dataset.img_size))

            mAP = 0
            #if epoch >= 20:
            #if epoch == 10 or epoch == 20 or epoch == 30:
            if epoch % 20 == 0:
                print('*'*20+"Validate"+'*'*20)
                self.inference()
                print("Start mAP")
                # calculate the AP after inferencing the evalset
                with torch.no_grad():
                    APs = self.evaluator.APs_run(self.cfg.EVAL["MULTI_SCALE_TEST"],self.cfg.EVAL["FLIP_TEST"] )    
                    valid_ap = 0
                    for cla in APs:
                        print("{} --> mAP : {}".format(cla, APs[cla]))
                        if (APs[cla] is not None) and (~np.isnan(APs[cla])) :   # actually nan should be 0 value
                            mAP += APs[cla]
                            valid_ap += 1
                    mAP = mAP / valid_ap
                    print('mAP:%g' % (mAP))
#########################################################################
            self.__save_model_weights(epoch, mAP)
            print('best mAP : %g' % (self.best_mAP))


    def inference(self):

        result_path =  self.cfg.EVAL_REUSLTS_DIR
        pred_cachedir = os.path.join(result_path, "cache")

        if os.path.exists(pred_cachedir):
            shutil.rmtree(pred_cachedir)  # delete the cache directory
        os.mkdir(pred_cachedir)

        imgs = os.listdir(self.img_valid_dir)
        visual_imgs = 0
        for v in imgs:
            path = os.path.join(self.img_valid_dir, v)
            img = cv2.imread(path)
            if img is None:
                print(f'****Could not read an image****: {path}')
                print('****Continue to the next image****')
                continue

            bboxes_prd = self.evaluator.get_bbox(img, self.cfg.EVAL["MULTI_SCALE_TEST"], self.cfg.EVAL["FLIP_TEST"]) 

            if bboxes_prd.shape[0] != 0 and visual_imgs < 30:   # 100
                boxes = bboxes_prd[..., :4]
                class_inds = bboxes_prd[..., 5].astype(np.int32)
                scores = bboxes_prd[..., 4]
                # this will not draw the boxes that have confidence score < 0.5  >> fixed 
                visualize_boxes(image=img, boxes=boxes, labels=class_inds, probs=scores, class_labels=self.cfg.DATA["CLASSES"])
                
                if cfg.EVAL['SHOW_RESULT']:
                    cv2.imshow('test_result', img) 
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()

                if cfg.EVAL['SAVE_RESULT']:
                    save_file = os.path.join(result_path, "{}".format(v))
                    cv2.imwrite(save_file, img)
                    # print("saved images : {}".format(save_file))
                visual_imgs += 1

            # write predict result in the text files
            for bbox in bboxes_prd:
                coor = np.array(bbox[:4], dtype=np.int32)
                score = bbox[4]
                class_ind = int(bbox[5])

                class_name =  self.cfg.DATA["CLASSES"][class_ind]
                score = '%.4f' % score
                xmin, ymin, xmax, ymax = map(str, coor)
                s = ' '.join([v, score, xmin, ymin, xmax, ymax]) + '\n'
                
                go = False
                while not go:
                    try:
                        with open(os.path.join(pred_cachedir, 'det_result_' + class_name + '.txt'), 'a') as f:
                            f.write(s)
                        go = True
                    except Exception as e:
                        print("Error massage:", e,)
                        print("wait for 2 seconds and try to write it again")
                        time.sleep(2)

        print("Finished evaluating")

if __name__ == "__main__":
    import config.yolov3_config_yoloformat as cfg
    parser = argparse.ArgumentParser()
    parser.add_argument('--weight_path', type=str, default='./weight/pre_weight_voc_repA1g4/best.pt', help='weight file path to load') # use  '' to train from scarch ./weight/12Fed2021-1/best.pt
    parser.add_argument('--resume', action='store_true',default=False,  help='resume training flag')
    parser.add_argument('--save_weight_dir', type=str, default='./weight', help='weight file path to save')
    parser.add_argument('--train_dir', type=str, default=f'./coco128/train/', help='train directory')
    parser.add_argument('--valid_dir', type=str, default=f'./coco128/valid/', help='valid directory')
    parser.add_argument('--gpu_id', type=int, default=0, help='gpu id')
    opt = parser.parse_args()

    Trainer(cfg=cfg,
            train_dir=opt.train_dir,
            valid_dir=opt.valid_dir,
            weight_path=opt.weight_path,
            resume=opt.resume,
            save_weight_dir=opt.save_weight_dir,
            gpu_id=opt.gpu_id).train()