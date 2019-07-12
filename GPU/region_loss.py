import time
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from utils import *

def build_targets(pred_boxes, target, anchors, num_anchors, nH, nW, noobject_scale, object_scale, sil_thresh, seen):
    nB = target.size(0)
    nA = num_anchors
    anchor_step = int(len(anchors)/num_anchors)
    conf_mask  = torch.ones(nB, nA, nH, nW) * noobject_scale
    coord_mask = torch.zeros(nB, nA, nH, nW)
    tx         = torch.zeros(nB, nA, nH, nW) 
    ty         = torch.zeros(nB, nA, nH, nW) 
    tw         = torch.zeros(nB, nA, nH, nW) 
    th         = torch.zeros(nB, nA, nH, nW) 
    tconf      = torch.zeros(nB, nA, nH, nW)

    nAnchors = nA*nH*nW
    nPixels  = nH*nW
    nGT = 0
    nCorrect = 0
    for b in range(nB):
        cur_pred_boxes = pred_boxes[b*nAnchors:(b+1)*nAnchors].t()
        cur_ious = torch.zeros(nAnchors)
        for t in range(50):
            if target[b][t*5+1] == 0:
                break
            gx = target[b][t*5+1]*nW
            gy = target[b][t*5+2]*nH
            gw = target[b][t*5+3]*nW
            gh = target[b][t*5+4]*nH
            cur_gt_boxes = torch.FloatTensor([gx,gy,gw,gh]).repeat(nAnchors,1).t()
            cur_ious = torch.max(cur_ious, bbox_ious(cur_pred_boxes, cur_gt_boxes, x1y1x2y2=False))
        cur_ious =  cur_ious.view(nA, nH, nW)   
        conf_mask[b][cur_ious>sil_thresh] = 0

    for b in range(nB):
        for t in range(50):
            if target[b][t*5+1] == 0:
                break
            nGT = nGT + 1
            best_iou = 0.0
            best_n = -1
            min_dist = 10000
            gx = target[b][t*5+1] * nW
            gy = target[b][t*5+2] * nH
            gi = int(gx)
            gj = int(gy)
            gw = target[b][t*5+3]*nW
            gh = target[b][t*5+4]*nH
            gt_box = [0, 0, gw, gh]
            for n in range(nA):
                aw = anchors[anchor_step*n]
                ah = anchors[anchor_step*n+1]
                anchor_box = [0, 0, aw, ah]
                iou  = bbox_iou(anchor_box, gt_box, x1y1x2y2=False)
                if anchor_step == 4:
                    ax = anchors[anchor_step*n+2]
                    ay = anchors[anchor_step*n+3]
                    dist = pow(((gi+ax) - gx), 2) + pow(((gj+ay) - gy), 2)
                if iou > best_iou:
                    best_iou = iou
                    best_n = n
                elif anchor_step==4 and iou == best_iou and dist < min_dist:
                    best_iou = iou
                    best_n = n
                    min_dist = dist

            gt_box = [gx, gy, gw, gh]
            pred_box = pred_boxes[b*nAnchors+best_n*nPixels+gj*nW+gi]

            coord_mask[b][best_n][gj][gi] = 1
            conf_mask[b][best_n][gj][gi] = object_scale
            tx[b][best_n][gj][gi] = target[b][t*5+1] * nW - gi
            ty[b][best_n][gj][gi] = target[b][t*5+2] * nH - gj
            tw[b][best_n][gj][gi] = math.log(gw/anchors[anchor_step*best_n])
            th[b][best_n][gj][gi] = math.log(gh/anchors[anchor_step*best_n+1])
            iou = bbox_iou(gt_box, pred_box, x1y1x2y2=False) # best_iou
            tconf[b][best_n][gj][gi] = iou
            if iou > 0.5:
                nCorrect = nCorrect + 1
    return nGT, nCorrect, coord_mask, conf_mask, tx, ty, tw, th, tconf


class RegionLoss(nn.Module):
    def __init__(self, anchors=[], num_anchors=1):
        super(RegionLoss, self).__init__()
        self.anchors = anchors
        self.num_anchors = num_anchors
        self.anchor_step = len(anchors)/num_anchors
        self.coord_scale = 1
        self.noobject_scale = 1
        self.object_scale = 5
        self.thresh = 0.6
        self.seen = 0

    def forward(self, output, target):
       
        nB = output.data.size(0)
        nA = self.num_anchors
        nH = output.data.size(2)
        nW = output.data.size(3)

        output   = output.view(nB, nA, (5), nH, nW)
        x    = F.sigmoid(output.index_select(2, Variable(torch.cuda.LongTensor([0]))).view(nB, nA, nH, nW))
        y    = F.sigmoid(output.index_select(2, Variable(torch.cuda.LongTensor([1]))).view(nB, nA, nH, nW))
        w    = output.index_select(2, Variable(torch.cuda.LongTensor([2]))).view(nB, nA, nH, nW)
        h    = output.index_select(2, Variable(torch.cuda.LongTensor([3]))).view(nB, nA, nH, nW)
        conf = F.sigmoid(output.index_select(2, Variable(torch.cuda.LongTensor([4]))).view(nB, nA, nH, nW))
        
        
        pred_boxes = torch.cuda.FloatTensor(4, nB*nA*nH*nW)
        grid_x = torch.linspace(0, nW-1, nW).repeat(nH,1).repeat(nB*nA, 1, 1).view(nB*nA*nH*nW).cuda()
        
        grid_y = torch.linspace(0, nH-1, nH).repeat(nW,1).t().repeat(nB*nA, 1, 1).view(nB*nA*nH*nW).cuda()
        anchor_w = torch.Tensor(self.anchors).view(nA, self.anchor_step).index_select(1, torch.LongTensor([0])).cuda()
        anchor_h = torch.Tensor(self.anchors).view(nA, self.anchor_step).index_select(1, torch.LongTensor([1])).cuda()
        anchor_w = anchor_w.repeat(nB, 1).repeat(1, 1, nH*nW).view(nB*nA*nH*nW)
        anchor_h = anchor_h.repeat(nB, 1).repeat(1, 1, nH*nW).view(nB*nA*nH*nW)
        
        
        
        pred_boxes[0] = x.data.view(nB*nA*nH*nW) + grid_x
        pred_boxes[1] = y.data.view(nB*nA*nH*nW) + grid_y
        pred_boxes[2] = torch.exp(w.data).view(nB*nA*nH*nW) * anchor_w
        pred_boxes[3] = torch.exp(h.data).view(nB*nA*nH*nW) * anchor_h
        pred_boxes = convert2cpu(pred_boxes.transpose(0,1).contiguous().view(-1,4))
        nGT, nCorrect, coord_mask, conf_mask, tx, ty, tw, th, tconf = build_targets(pred_boxes, target.data, self.anchors, nA, \
                                                               nH, nW, self.noobject_scale, self.object_scale, self.thresh, self.seen)
        
        tx    = Variable(tx.cuda())
        ty    = Variable(ty.cuda())
        tw    = Variable(tw.cuda())
        th    = Variable(th.cuda())
        tconf = Variable(tconf.cuda())
        coord_mask = Variable(coord_mask.cuda())
        conf_mask  = Variable(conf_mask.cuda().sqrt())
        
        loss_x = self.coord_scale * nn.MSELoss(size_average=False)(x*coord_mask, tx*coord_mask)/2.0
        loss_y = self.coord_scale * nn.MSELoss(size_average=False)(y*coord_mask, ty*coord_mask)/2.0
        loss_w = self.coord_scale * nn.MSELoss(size_average=False)(w*coord_mask, tw*coord_mask)/2.0
        loss_h = self.coord_scale * nn.MSELoss(size_average=False)(h*coord_mask, th*coord_mask)/2.0
        loss_conf = nn.MSELoss(size_average=False)(conf*conf_mask, tconf*conf_mask)/2.0

        loss = loss_x + loss_y + loss_w + loss_h + loss_conf

                
        print('%d: nGT %d, recall %d,  loss: x %f, y %f, w %f, h %f, conf %f, total %f' % (self.seen, nGT, nCorrect, loss_x.data[0], loss_y.data[0], loss_w.data[0], loss_h.data[0], loss_conf.data[0],  loss.data[0]))

        return loss