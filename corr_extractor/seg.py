import torch.nn as nn
import torch.nn.functional as F
from utils import *

class PoseSegLayer(nn.Module):
    def __init__(self, options):
        super(PoseSegLayer, self).__init__()
        self.num_classes = int(options['classes'])
        self.training = False
        self.pnp = False

    def forward(self, output):

        # output : nB x nC x H x W
        # output : BxAs*(1+2*num_vpoints+num_classes)*H*W
        t0 = time.time()
        nB = output.data.size(0)
        nA = 1
        nC = self.num_classes
        nH = output.data.size(2)
        nW = output.data.size(3)

        # update object_scale according to nA and nH and nW
        # self.object_scale = nA * nH * nW * 0.01

        output = output.view(nB * nA, (nC), nH * nW).transpose(0, 1). \
            contiguous().view((nC), nB * nA * nH * nW)

        cls = output[0:nC].transpose(0, 1)
        t1 = time.time()

        if self.training or self.pnp:
            output = output.transpose(0, 1)
            return output
        else:
            cls_confs, cls_ids = torch.max(F.softmax(cls, 1), 1) # not quite understand
            cls_confs = cls_confs.view(nB, nH, nW)
            cls_ids = cls_ids.view(nB, nH, nW)

            # copy to CPU
            cls_confs = convert2cpu(cls_confs).detach().numpy()
            cls_ids = convert2cpu_long(cls_ids).detach().numpy()

            t2 = time.time()

            show_time = False
            if show_time:
                print('---------------------------------')
                print('matrix computation : %f' % (t1 - t0))
                print('        gpu to cpu : %f' % (t2 - t1))
                print('---------------------------------')

            out_preds = [cls_confs, cls_ids]
            return out_preds