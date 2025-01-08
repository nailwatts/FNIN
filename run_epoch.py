import torch
from training_utils.make_scales import make_scales
from data_processing.depth_to_normals import get_normals_from_depth_list, get_gradient_from_depth_list
import time


def prepare_gt(sample, device, multi_scale=True):
	gt = {}
	if multi_scale:
		gt['mask_scales'] = make_scales(sample['mask'].float().to(device))
	else:
		gt['mask_scales'] = [sample['mask'].float().to(device)]
	if 'depth' in sample:
		if multi_scale:
			gt['depth_scales'] = make_scales(sample['depth'].float().to(device))
		else:
			gt['depth_scales'] = [sample['depth'].float().to(device)]
		if 'n_intrinsics' in sample.keys():
			gt['nfd_scales'] = get_normals_from_depth_list(gt['depth_scales'],sample['n_intrinsics'])
		gt['gradient_scales'] = get_gradient_from_depth_list(gt['depth_scales'])

	if 'n_intrinsics' in sample.keys():
		gt['n_intrinsics'] = sample['n_intrinsics'].to(device)
	gt['mean_distance'] = sample['mean_distance'].to(device)
	return gt


class PreprocessSynthetic:
	def __init__(self, device, image_augmentations=[]):
		self.device = device
		self.image_augmentations = image_augmentations

	def __call__(self, sample):
		for k in ['depth', 'normal', 'intrinsics','mask','mean_distance','n_intrinsics', 'f']:
			if k in sample.keys():
				sample[k] = sample[k].to(self.device)

		for aug in self.image_augmentations:
			aug(sample['normal'])
		
		return sample
		
class PreprocessReal:
	def __init__(self, device):
		self.device = device

	def __call__(self, sample):
		for k in ['depth', 'normal','intrinsics','mean_distance','mask','n_intrinsics']:
			if k in sample.keys():
				sample[k] = sample[k].to(self.device)

		return sample
		
def train_epoch(net, dataloader, sample_preprocessing_fun, device, criterion, optimizer, scheduler, scalar_logger=None, image_logger=None, multi_scale=True):
	
	net.train()

	for batch_num, sample in enumerate(dataloader):
		
		sample = sample_preprocessing_fun(sample)
		target = prepare_gt(sample, device, multi_scale)



		optimizer.zero_grad()
		output = net(sample['normal'], sample['n_intrinsics'], sample['mask'])
		loss,losses_dict = criterion(output,target)
		
		loss.backward()
		optimizer.step()

			
		
		if scalar_logger is not None:
			scalar_logger(losses_dict,[0])
			
		if image_logger is not None:
			log_images = {}
			for k,v in output.items():
				log_images['output_' + k] = v
			for k,v in target.items():
				log_images['target_' + k] = v 
			log_images['normal'] = sample['normal']
			log_images['error_map'] = torch.abs(sample['depth']*sample['mean_distance'].view(-1,1,1,1) - output['depth_scales'][-1]*sample['mean_distance'].view(-1,1,1,1)) / 5

			image_logger(log_images, sample['name'][0], sample['mask'])
	scheduler.step()
	print("learning rate is ", optimizer.param_groups[0]["lr"])

			
def eval_epoch(net, dataloader, sample_preprocessing_fun, device, criterion=None,  scalar_logger=None, image_logger=None, test_mode=True, log_meshes=False, multi_scale=True):
	net.eval()
	total_time = 0
	for batch_num, sample in enumerate(dataloader):
		sample = sample_preprocessing_fun(sample)
		start = time.time()
		intrinsics = None
		if 'n_intrinsics' in sample.keys():
			intrinsics = sample['n_intrinsics']

		output = net(sample['normal'], intrinsics, sample['mask'], test_mode=test_mode)
		end = time.time()
		total_time += (end - start)

		if criterion is not None:
			target = prepare_gt(sample, device, multi_scale)
			loss,losses_dict = criterion(output,target)
		if scalar_logger is not None:
			scalar_logger(losses_dict,sample['name'][0])
			
		if image_logger is not None:
			log_images = {}
			for k,v in output.items():
				log_images['output_' + k] = v
			for k,v in target.items():
				log_images['target_' + k] = v
			log_images['normal'] = sample['normal']

			log_images['error_map'] = torch.abs(sample['depth']*sample['mean_distance'].view(-1,1,1,1) - output['depth_scales'][-1]*sample['mean_distance'].view(-1,1,1,1)) / 5

			if log_meshes:
				mesh_data= {}
				mesh_data['mask'] = sample['mask'].to('cpu')
				if 'n_intrinsics' in sample.keys():
					mesh_data['intrinsics'] = sample['n_intrinsics'].to('cpu')
				else:
					mesh_data['intrinsics'] = sample['intrinsics'].to('cpu')
				mesh_data['net_depth'] = output['depth_scales'][-1].to('cpu')
			else:
				mesh_data=None
			image_logger(log_images, sample['name'][0], sample['mask'], mesh_data=mesh_data)

	print(total_time / 14)
			
			
