import torch.optim as optim
from models.criterion import *
from models.multiscale_net import *
from models.resnet import *
from models.unet import *
from models.fno import *
from models.resformer import *
from models.dptnet import *
from models.multiscale_fno import *
from datasets.real_dataset import RealDataset
from models.save_load_checkpoint import *
from data_processing.common_transforms import *
from run_epoch import eval_epoch, PreprocessReal
from training_utils.logger import *
import os
import argparse

os.environ['CUDA_VISIBLE_DEVICES'] = "1"


parser = argparse.ArgumentParser()

parser.add_argument('logdir', default=None, help='the path to store logging information and models and models')
parser.add_argument('--gpu', default=False,  action='store_true', help='enable to run on gpu')
parser.add_argument('--net_arc', type=str, help='network architecture', default='resnet')

parser.add_argument('--real_dataset_root', type=str, help='path to real dataset', default='')
parser.add_argument('--batch_size', type=int, default=1, help='input batch size')
# The training weight 
parser.add_argument('--depth_weight', type=float, default=1.0, help='the weight for the diffuse component')
parser.add_argument('--normal_weight', type=float, default=1.0, help='the weight for the diffuse component')
parser.add_argument('--nfd_weight', type=float, default=1.0, help='the weight for the roughness component')
parser.add_argument('--gradient_weight', type=float, default=1.0, help='the weight for the roughness component')
#logging options
parser.add_argument('--scalars_to_log', type=list, default=['loss','depth_scales_loss','depth_scales_loss_scale','abs_depth_loss_scale', 'l1_depth_loss_scales'], help='the scalars to log')
parser.add_argument('--image_logger_keys', type=list, default=['output_depth_scales', 'output_attention_scales']) # output_attention_scales
parser.add_argument('--test_scalar_lf', type=int, default=1, help='frequency to log scalars during testing')
parser.add_argument('--test_image_lf', type=int, default=1, help='frequency to log images during testing')
parser.add_argument('--save_first_batch_only', default=True,  action='store_true', help='only save outputs at largest scale')
#checkpoints tp load
parser.add_argument('--checkpoint', default='None', help='path to checkpoint to load')
parser.add_argument('--uncalibrated',default=False, action='store_true', help='use calibration network')
parser.add_argument('--calib_net_checkpoint', type=str)
#mesh logging
parser.add_argument('--post_integrate_normals', default=False, action='store_false')
parser.add_argument('--log_meshes', default=True, action='store_false')


opt = parser.parse_args()
if opt.gpu:
	device = 'cuda'
else:
	if torch.cuda.is_available():
		import warnings
		warnings.warn('running on CPU but GPUs detected. Add arg \"--gpu\" to run on gpu')
	device='cpu'



real_test_data = RealDataset(opt.real_dataset_root, transform=real_transforms)
real_test_loader = torch.utils.data.DataLoader(real_test_data, batch_size=1, shuffle=False, num_workers=1)
real_preprocessing_fun = PreprocessReal(device)


#setup network
if opt.net_arc == 'resnet':
	net = ResNet()
	multi_scale = False
elif opt.net_arc == 'unet':
	net = UNet()
	multi_scale = False
elif opt.net_arc == 'resformer':
	net = ResFormer()
	multi_scale = False
elif opt.net_arc == 'fno':
	net = FNO()
	multi_scale = False
elif opt.net_arc == 'dpt':
	net = dpt()
	multi_scale = False
elif opt.net_arc == 'multiscale_fno':
	net = MultiscaleFNO()
	multi_scale = True
elif opt.net_arc == 'multiscale':
	net = MultiscaleNet()
	multi_scale = True
net.to(device)
load_checkpoint(opt.checkpoint, net=net)


if opt.uncalibrated:
	calib_net = CalibrationNet(batch_norm=True)
	calib_net.to(device)
	load_checkpoint(opt.calib_net_checkpoint, net=calib_net)
else:
	calib_net = None

if opt.gpu:
	net = nn.DataParallel(net)

# criterion = Criterion(opt.depth_weight, opt.normal_weight, opt.nfd_weight, opt.gradient_weight)

#make logdir
if not os.path.exists(opt.logdir):
	os.mkdir(opt.logdir)
test_image_dir = os.path.join(opt.logdir,'images')
if not os.path.exists(test_image_dir):
	os.mkdir(test_image_dir)

# scalar_logger = ScalarLogger(os.path.join(opt.logdir,'eval_log_diligent.txt'), log_freq=1, keys=opt.scalars_to_log)
image_logger = ImageLogger(test_image_dir,log_freq=1,save_first_batch_only=opt.save_first_batch_only,keys=opt.image_logger_keys)
with torch.no_grad():
	eval_epoch(net, real_test_loader, real_preprocessing_fun, device,  criterion=None, scalar_logger=None, image_logger=image_logger,calibration_net=calib_net, post_integrate_normals=opt.post_integrate_normals, log_meshes=opt.log_meshes, multi_scale=multi_scale)
# scalar_logger.summarize()
	
	
	
	
	
