import sys
sys.path.append("..")

import os
# AbsolutePath = os.path.abspath(__file__)           
# SuperiorCatalogue = os.path.dirname(AbsolutePath)   
# BaseDir = os.path.dirname(SuperiorCatalogue)       
# sys.path.insert(0,BaseDir)                          

import torch.nn as nn
import torch
from model.backbones.darknet53 import MobilnetV2, RepVGG, Darknet53
from model.necks.yolo_fpn import FPN_YOLOV3
from model.head.yolo_head import Yolo_head
from model.layers.conv_module import Convolutional
import numpy as np
from utils.tools import *

import config.yolov3_config_voc as cfg


class Yolov3(nn.Module):
    """
    Note ： int the __init__(), to define the modules should be in order, because of the weight file is order
    """
    optional_groupwise_layers = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26]
    g2_map = {l: 2 for l in optional_groupwise_layers}
    g4_map = {l: 4 for l in optional_groupwise_layers}
    g8_map = {l: 8 for l in optional_groupwise_layers}
    g16_map = {l: 16 for l in optional_groupwise_layers}
    g32_map = {l: 32 for l in optional_groupwise_layers}
    A_num_blocks = [2, 4, 14, 1]
    B_num_blocks = [4, 6, 16, 1]
    w_A0 = [0.75, 0.75, 0.75, 2.5] #[1280, 192, 96]
    w_A1 = [1, 1, 1, 2.5]  # [1280, 256, 128]
    
    def __init__(self, init_weights=True, deploy=False):
        super(Yolov3, self).__init__()

        self.__anchors = torch.FloatTensor(cfg.MODEL["ANCHORS"])
        self.__strides = torch.FloatTensor(cfg.MODEL["STRIDES"])
        self.__nC = cfg.DATA["NUM"]
        self.__out_channel = cfg.MODEL["ANCHORS_PER_SCLAE"] * (self.__nC + 5)

        # self.__backbone = MobilnetV2()
        # #self.__fpn = FPN_YOLOV3(fileters_in=[1024, 512, 256],
        # #                        fileters_out=[self.__out_channel, self.__out_channel, self.__out_channel])
        # self.__fpn = FPN_YOLOV3(fileters_in=[320, 96, 32],
        #                         fileters_out=[self.__out_channel, self.__out_channel, self.__out_channel])

        ##################################################################################################
        self.__backbone = RepVGG(num_blocks=self.A_num_blocks, width_multiplier=self.w_A1, 
                                 override_groups_map=self.g4_map, deploy=deploy)
        self.__fpn = FPN_YOLOV3(fileters_in=[1280, 256, 128],
                                fileters_out=[self.__out_channel, self.__out_channel, self.__out_channel])
        ##################################################################################################
        self.__backbone = Darknet53()
        self.__fpn = FPN_YOLOV3(fileters_in=[1024, 512, 256],
                                fileters_out=[self.__out_channel, self.__out_channel, self.__out_channel])
        ###################################################################################################

        # small
        self.__head_s = Yolo_head(nC=self.__nC, anchors=self.__anchors[0], stride=self.__strides[0])
        # medium
        self.__head_m = Yolo_head(nC=self.__nC, anchors=self.__anchors[1], stride=self.__strides[1])
        # large
        self.__head_l = Yolo_head(nC=self.__nC, anchors=self.__anchors[2], stride=self.__strides[2])

        if init_weights:
            self.__init_weights()


    def forward(self, x):
        out = []

        x_s, x_m, x_l = self.__backbone(x)
        x_s, x_m, x_l = self.__fpn(x_l, x_m, x_s)

        out.append(self.__head_s(x_s))
        out.append(self.__head_m(x_m))
        out.append(self.__head_l(x_l))

        if self.training:
            p, p_d = list(zip(*out))
            return p, p_d  # small, medium, large list of [(bs,nG,nG,nA,8),(bs,nG,nG,nA,8), (bs,nG,nG,nA,8)] small(1/8) medium large
        else:
            p, p_d = list(zip(*out))
            return p, torch.cat(p_d, 0)   # a list of [(all boxes,8), (all boxes,8), (all boxes,8)] >> so torch.cat(p_d, 0) = (total boxes, 8)
        # boxes are in xywh

    def __init_weights(self):

        " Note ：nn.Conv2d nn.BatchNorm2d'initing modes are uniform "
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.normal_(m.weight.data, 0.0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()
                print("initing {}".format(m))

            elif isinstance(m, nn.BatchNorm2d):
                torch.nn.init.constant_(m.weight.data, 1.0)
                torch.nn.init.constant_(m.bias.data, 0.0)

                print("initing {}".format(m))


    def load_preweight_mobilev2(self):
        mobilenet = torchvision.models.mobilenet_v2(pretrained=True)
        our_state_dict = self.module_list.state_dict()
        for (old_k, old_p) , (pre_k, pre_p) in zip(self.module_list[0:18].state_dict().items(), mobilenet.features[0:18].state_dict().items()):
            our_state_dict.update({old_k: pre_p})
        self.module_list.load_state_dict(our_state_dict, strict=False)

    def load_weight_mobilev2(self, path):
        state_dict = torch.load(weightfile)
        self.load_state_dict(state_dict)


if __name__ == '__main__':
    weightfile = 'weight/lw_movb2_fp_nopretrain03Feb2021/best.pt'

    net = Yolov3()
    print(net)
    print(net._Yolov3__backnone.conv1.weight[0,0,0])
    # for name, param in net.state_dict().items():
    #     print(name)
    # print(net.state_dict())
    net.load_weight_mobilev2(weightfile)
    print(net._Yolov3__backnone.conv1.weight[0,0,0])

    net.eval()
    in_img = torch.randn(12, 3, 416, 416)
    p, p_d = net(in_img)

    if net.training:
        for i in range(3):
            print(p[i].shape)
            print(p_d[i].shape)
    else:
        for i in range(3):
            print(p[i].shape)
        print(p_d.shape)