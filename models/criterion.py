import torch
import torch.nn as nn
from training_utils import loss_functions
from data_processing.depth_to_normals import get_normals_from_depth_list, get_gradient_from_depth_list

class Criterion(nn.Module):
	def __init__(self, depth_weight, normal_weight, nfd_weight, gradient_weight):
		super(Criterion,self).__init__()
		self.key_weight = {'depth_scales': depth_weight, 'normal_scales': normal_weight, 'nfd_scales': nfd_weight, 'gradient_scales': gradient_weight}

	def list_loss(self,a_s,b_s,masks, attentions, base_loss_function):
		losses = []
		loss = 0
		if attentions is None:
			attentions = [torch.ones_like(a) for a in a_s]
		for a, b, mask, attention in zip(a_s,b_s,masks, attentions):
			loss_i = base_loss_function(a, b, mask, attention)
			loss = loss + loss_i
			losses.append(loss_i)

		return loss, losses

	def forward(self, output, target):
		loss = 0
		loss_dict = {}
		target_keys = target.keys()

		if 'n_intrinsics' in target_keys:
			output['nfd_scales'] = get_normals_from_depth_list(output['depth_scales'],target['n_intrinsics'])
		output['gradient_scales'] = get_gradient_from_depth_list(output['depth_scales'])
        	
		for key in ['depth_scales', 'nfd_scales']:
			attention = None
			if key in target_keys:
				if 'attention_scales' in output.keys():
					# the attention used here must be consistent with the weight in multiscale_fno.py
					if key == 'depth_scales':
						attention = [1 - attention_map for attention_map in output['attention_scales']]
					else:
						attention = [torch.cat((attention_map, attention_map, attention_map), dim=1) for attention_map in output['attention_scales']]

				key_loss, key_loss_scales = self.list_loss(target[key],output[key],target['mask_scales'], attention, loss_functions.L1_loss)
				loss += self.key_weight[key]*key_loss
				loss_dict[key + '_loss'] = key_loss.item()
				loss_dict[key + '_loss_scale'] = [x.item() for x in key_loss_scales]
			
		with torch.no_grad():
			if 'mean_distance' in target_keys:
				attention = None
				gt_depths_abs = [target['mean_distance'].view(-1,1,1,1)*x for x in target['depth_scales']]
				output_depths_abs = [target['mean_distance'].view(-1, 1, 1, 1) * x for x in output['depth_scales']]
				_, abs_depth_loss_scales = self.list_loss(gt_depths_abs, output['depth_scales'], target['mask_scales'], attention, loss_functions.L1_loss_scale_inv)
				loss_dict['abs_depth_loss_scale'] = [x.item() for x in abs_depth_loss_scales]
				_, l1_depth_loss_scales = self.list_loss(gt_depths_abs, output_depths_abs, target['mask_scales'], attention,
														  loss_functions.L1_loss_trans_inv)
				loss_dict['l1_depth_loss_scales'] = [x.item() for x in l1_depth_loss_scales]

         		
			if 'int_depth' in output:
				gt_depth_abs = target['mean_distance'].view(-1,1,1,1)*target['depth_scales'][-1]
				abs_depth_loss_int = loss_functions.L1_loss_scale_inv(gt_depth_abs, output['int_depth'].to(gt_depth_abs.device), target['mask_scales'][-1])
				loss_dict['abs_depth_loss_int'] = abs_depth_loss_int.item()
         			
		if isinstance(loss, torch.Tensor):
			loss_dict['loss'] = loss.item()
		else:
			loss_dict['loss'] = loss
		return loss, loss_dict
     
