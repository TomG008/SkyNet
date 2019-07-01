import cv2
from utils import *
from models import *
import torch
from torchvision import transforms
import argparse

modeltype = 'SkyNet()'
weightfile = 'dac.weights'
model = eval(modeltype)
region_loss = model.loss
load_net(weightfile, model)
region_loss.seen = model.seen

model = model.cuda()

model.eval()
cur_model = model

anchors = cur_model.anchors
num_anchors = cur_model.num_anchors
anchor_step = len(anchors) // num_anchors
h = 20
w = 40
batch_size = 1

grid_x = torch.linspace(0, w - 1, w).repeat(h, 1).repeat(batch_size * num_anchors, 1, 1).view(
    batch_size * num_anchors * h * w).cuda()
grid_y = torch.linspace(0, h - 1, h).repeat(w, 1).t().repeat(batch_size * num_anchors, 1, 1).view(
    batch_size * num_anchors * h * w).cuda()
anchor_w = torch.Tensor(anchors).view(num_anchors, anchor_step).index_select(1, torch.LongTensor([0]))
anchor_h = torch.Tensor(anchors).view(num_anchors, anchor_step).index_select(1, torch.LongTensor([1]))
anchor_w = anchor_w.repeat(batch_size, 1).repeat(1, 1, h * w).view(batch_size * num_anchors * h * w).cuda()
anchor_h = anchor_h.repeat(batch_size, 1).repeat(1, 1, h * w).view(batch_size * num_anchors * h * w).cuda()
sz_hw = h * w
sz_hwa = sz_hw * num_anchors

e_shape = model.width, model.height
trans = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.25, 0.25, 0.25]),
])


def solve(raw_img):
    img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    img = img.resize(e_shape)
    img = trans(img)
    img = img.view(1, 3, 160, 320)
    img = img.cuda()
    output = model(img).data
    batch = output.size(0)
    output = output.view(batch * num_anchors, 5, h * w).transpose(0, 1).contiguous().view(5,
                                                                                          batch * num_anchors * h * w)

    det_confs = torch.sigmoid(output[4])
    det_confs = convert2cpu(det_confs)

    for b in range(batch):
        det_confs_inb = det_confs[b * sz_hwa:(b + 1) * sz_hwa].numpy()
        ind = np.argmax(det_confs_inb)

        xs_inb = torch.sigmoid(output[0, b * sz_hwa + ind]) + grid_x[b * sz_hwa + ind]
        ys_inb = torch.sigmoid(output[1, b * sz_hwa + ind]) + grid_y[b * sz_hwa + ind]
        ws_inb = torch.exp(output[2, b * sz_hwa + ind]) * anchor_w[b * sz_hwa + ind]
        hs_inb = torch.exp(output[3, b * sz_hwa + ind]) * anchor_h[b * sz_hwa + ind]

        bcx = xs_inb.item() / w
        bcy = ys_inb.item() / h
        bw = ws_inb.item() / w
        bh = hs_inb.item() / h

        xmin = int((bcx - bw / 2.0) * raw_width)
        ymin = int((bcy - bh / 2.0) * raw_height)
        xmax = int(xmin + bw * raw_width)
        ymax = int(ymin + bh * raw_height)

        cv2.rectangle(raw_img, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)

    return raw_img


def main(args, record=False):
    global raw_width, raw_height, video

    if args.input:
        raw_img = cv2.imread(args.input)
        raw_width, raw_height = raw_img.shape[0], raw_img.shape[1]
        solve(raw_img)
        cv2.imshow('my webcam', raw_img)
        cv2.waitKey(0)
    else:
        cam = cv2.VideoCapture(0)
        raw_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
        raw_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if record:
            output_file = 'demo.avi'
            fourcc = cv2.VideoWriter_fourcc('I', '4', '2', '0')
            video = cv2.VideoWriter(output_file, fourcc, 60, (raw_width, raw_height))
        else:
            video = None
        mirror = True

        while True:
            ret_val, raw_img = cam.read()
            if mirror:
                raw_img = cv2.flip(raw_img, 1)

            raw_img = solve(raw_img)

            if cv2.waitKey(1) == 27:
                break

            cv2.imshow('my webcam', raw_img)
            if video is not None:
                video.write(raw_img)

    cv2.destroyAllWindows()

    if video is not None:
        video.release()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str)
    args = parser.parse_args()
    main(args, record=False)
