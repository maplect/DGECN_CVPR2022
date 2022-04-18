import imp
from matplotlib.backend_bases import MouseEvent

#from regex import A
from utils import *
from dgecn import dgecn
import cv2
import numpy as np
import argparse

def parse_args():
    parser = argparse.ArgumentParser(
        description='Test on YCB-V')
    parser.add_argument('--ext', type=str,
                        help='image extension to search for in folder', default="png")
    parser.add_argument('--test_mode', type=str,
                        help='test on YCB-Video or depth', default="YCB-Video")
    parser.add_argument('--out_dir', type=str,
                        help='output dir', default="./Result")
    parser.add_argument('--filelist', type=str,
                        help='file list for test', default="ycb-video-testlist.txt")
    parser.add_argument('--pretrained', type=str,
                        help='pretrained model', default="dgecn.pth")
    parser.add_argument("--gpus", type=str,
                        help='choose gpu',
                        default='0')
    parser.add_argument("--use_gpu",
                        help='if set, use gpu).',
                        action='store_true')

    return parser.parse_args() 

def evaluate_all(data_cfg, weightfile, listfile, outdir, object_names, intrinsics, vertex,
                         bestCnt, conf_thresh, linemod_index=False, use_gpu=False, gpu_id='0'):
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data_options = read_data_cfg(data_cfg)

    model = dgecn(data_options)
    model = torch.load(weightfile)
    print('Loading weights from %s... Done!' % (weightfile))

    if use_gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
        model.cuda()

    with open(listfile, 'r') as file:
        imglines = file.readlines()

    for idx in range(len(imglines)):
        imgfile = imglines[idx].rstrip()
        img = cv2.imread(imgfile)
        dirname, filename = os.path.split(imgfile)
        baseName, _ = os.path.splitext(filename)
        if linemod_index:
            outFileName = baseName[-4:]
        else:
            dirname = os.path.splitext(dirname[dirname.rfind('/') + 1:])[0]
            outFileName = dirname+'_'+baseName

        start = time.time()
        predPose = do_detect_all(model, img, intrinsics, bestCnt, conf_thresh, use_gpu)
        finish = time.time()

        arch = 'CPU'
        if use_gpu:
            arch = 'GPU'
        print('%s: Predict %d objects in %f seconds (on %s).' % (imgfile, len(predPose), (finish - start), arch))
        print("Prediction saved!", outFileName, predPose, outdir)
        save_predictions(outFileName, predPose, object_names, outdir)

        # visualize predictions
        vis_start = time.time()
        visImg = visualize_predictions(predPose, img, vertex, intrinsics)
        cv2.imwrite(outdir + '/' + outFileName + '.jpg', visImg)
        print("Results saved as :", outdir + '/' + outFileName + '.jpg')
        vis_finish = time.time()
        print('%s: Visualization in %f seconds.' % (imgfile, (vis_finish - vis_start)))



if __name__ == '__main__':
    args = parse_args()
    use_gpu = args.use_gpu
    test = args.test_mode
    outdir = args.out_dir
    filelist = args.filelist
    model_path = args.pretrained
    gpu_id = args.gpus

    """ if test == 'depth':
        print("test on depth")
        evaluate_depth('./data/data-YCB.cfg',
                     model_path,
                     filelist,
                     outdir, use_gpu=use_gpu)
 """


    if test == 'YCB-Video':
        print("test on YCB-Video")
        # intrinsics of YCB-VIDEO dataset
        k_ycbvideo = np.array([[1.06677800e+03, 0.00000000e+00, 3.12986900e+02],
                               [0.00000000e+00, 1.06748700e+03, 2.41310900e+02],
                               [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
        # 21 objects for YCB-Video dataset
        object_names_ycbvideo = ['002_master_chef_can', '003_cracker_box', '004_sugar_box', '005_tomato_soup_can', '006_mustard_bottle',
                                 '007_tuna_fish_can', '008_pudding_box', '009_gelatin_box', '010_potted_meat_can', '011_banana',
                                 '019_pitcher_base', '021_bleach_cleanser', '024_bowl', '025_mug', '035_power_drill', '036_wood_block',
                                 '037_scissors', '040_large_marker', '051_large_clamp', '052_extra_large_clamp', '061_foam_brick']
        vertex_ycbvideo = np.load('./data/YCB-Video/YCB_vertex.npy')
        evaluate_all('./data/data-YCB.cfg',
                     model_path,
                     filelist,
                     outdir, object_names_ycbvideo, k_ycbvideo, vertex_ycbvideo,
                     bestCnt=10, conf_thresh=0.3, use_gpu=use_gpu)
 
    else:
        print('unsupported dataset \'%s\'.' % test)
