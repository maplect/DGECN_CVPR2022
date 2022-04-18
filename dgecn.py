import imp
import torch
import torch.nn as nn
from backbone import Darknet
from corr_extractor import *
from utils import *
from depth_net import *
class dgecn(nn.Module):
    def __init__(self, data_options):
        super(dgecn, self).__init__()

        pose_arch_cfg = data_options['pose_arch_cfg']
        self.width = int(data_options['width'])
        self.height = int(data_options['height'])
        self.channels = int(data_options['channels'])

        # note you need to change this after modifying the network
        self.output_h = 76
        self.output_w = 76

        self.encoder = ResnetEncoder(18,False)
        self.depthlayer = DepthDecoder(num_ch_enc=self.encoder.num_ch_enc, scales=range(4))
        self.coreModel = Darknet(pose_arch_cfg, self.width, self.height, self.channels)
        self.segLayer = PoseSegLayer(data_options)
        self.regLayer = Pose2DLayer(data_options)
        self.training = False

    def forward(self, x, y = None):
        if self.training:
            feature = self.encoder(x)
            depth_pred = self.depthlayer(feature)
            disp = depth_pred[("disp", 0)]
            scaled_disp, depth = disp_to_depth(disp, 0.1, 100)
            outlayers = self.coreModel(x+depth[0][0])
            out1 = self.segLayer(outlayers[0])
            out2 = self.regLayer(outlayers[1])
            out_preds = [out1, out2]
            return out_preds
        else:
            feature = self.encoder(x)
            depth_pred = self.depthlayer(feature)
            disp = depth_pred[("disp", 0)]
            scaled_disp, depth = disp_to_depth(disp, 0.1, 100)
            disp_resized = torch.nn.functional.interpolate(
                disp, (480, 640), mode="bilinear", align_corners=False)
            
            outlayers = self.coreModel(x+depth[0][0])
            out1 = self.segLayer(outlayers[0])
            out2 = self.regLayer(outlayers[1])
            out_preds = [out1, out2]
            return out_preds, disp_resized

    def eval(self):
        self.encoder.eval()
        self.depthlayer.eval()
        self.coreModel.eval()
        self.segLayer.eval()
        self.regLayer.eval()
        self.training = False

    def load_weights(self, weightfile):
        #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.coreModel.load_state_dict(torch.load(weightfile))
        model_path = os.path.join("../DGECN/models", "depth")
        print("-> Loading model from ", model_path)
        encoder_path = os.path.join(model_path, "encoder.pth")
        depth_decoder_path = os.path.join(model_path, "depth.pth")

        # LOADING PRETRAINED MODEL
        print("   Loading pretrained encoder")
        #encoder = depth_net.ResnetEncoder(18, False)
        loaded_dict_enc = torch.load(encoder_path)

        # extract the height and width of image that this model was trained with
        feed_height = loaded_dict_enc['height']
        feed_width = loaded_dict_enc['width']
        filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in self.encoder.state_dict()}
        self.encoder.load_state_dict(filtered_dict_enc)
        loaded_dict = torch.load(depth_decoder_path)
        self.depthlayer.load_state_dict(loaded_dict)

if __name__ == '__main__':
    
    data_options = read_data_cfg('./data/data-YCB.cfg')
    m = dgecn(data_options)
    m.load_weights('../DGECN/models/expdepth30.pth')
    
    """lr = 1e-3
    optimizer = torch.optim.Adam(m.parameters(), lr=lr)"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('save pth')
    """ batch=1
    image = np.zeros((batch, m.width, m.height,3))
    img = torch.from_numpy(image.transpose(0, 3, 1, 2)).float().div(255.0)
    img = img.cuda()
    img = Variable(img)
    #m.cuda()
    m.eval()
    out = m(img)
    print(out[0][0].shape) """
    torch.save(m, 'dgecn.pth')
    
    print("SAVED")
