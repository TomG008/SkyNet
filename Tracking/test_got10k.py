# -*- coding: utf-8 -*
import matplotlib
matplotlib.use('Agg')
from got10k.trackers import Tracker
from got10k.experiments import *
from models.custom import Custom_Sky, Custom
import torch
from tools.test import siamese_init, siamese_track, load_pretrain, load_config
import numpy as np
import argparse
import cv2

parser = argparse.ArgumentParser(description='PyTorch Tracking Demo')
parser.add_argument('--config', dest='config', default='demo_res.json',
                    help='hyper-parameter of SiamMask in json format')
args = parser.parse_args()

class myTracker(Tracker):
    def __init__(self, model, ckpt, idx, args):
        super(myTracker, self).__init__(name='ResNet{0}'.format(idx))
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.cfg = load_config(args)
        torch.backends.cudnn.benchmark = True
        siammask = model(anchors=self.cfg['anchors'])
        siammask = load_pretrain(siammask, ckpt)
        siammask.eval().to(self.device)
        self.model = siammask
        self.box = None
        self.state = None

    def init(self, image, box):
        '''
        :param self:
        :param image:
        :param box: [0, 1, 0+2， 1+3]
        :return:
        '''
        image = np.asarray(image)
        self.box = box
        x, y = box[0] + box[2] / 2, box[1] + box[3] / 2
        w, h = box[2], box[3]
        target_pos = np.array([x, y])
        target_sz = np.array([w, h])
        self.state = siamese_init(image, target_pos, target_sz, self.model, self.cfg['hp'], device=self.device)

    def update(self, image):
        image = np.asarray(image)
        self.state = siamese_track(self.state, image, mask_enable=True, refine_enable=False, device=self.device)
        x, y = self.state['target_pos']
        w, h = self.state['target_sz']
        self.box = np.array([x - w / 2, y - h / 2, w, h])
        return self.box


if __name__ == '__main__':
    # setup tracker
    for idx in range(20):
        tracker = myTracker(model=Custom, ckpt='0902res/checkpoint_e{0}.pth'.format(idx + 1), idx=idx+1, args=args)

        # run experiments on GOT-10k (validation subset)
        experiment = ExperimentGOT10k('data/got10k', subset='test')
        experiment.run(tracker, visualize=False)

        # report performance
        experiment.report([tracker.name])
