import torch
import torch.nn.functional as F

def make_scales(image,normalized=False,mode='bilinear',min_size=64, pop=False):
	if normalized:
		image = F.normalize(image,dim=1,p=2)
		
	images = [image]
	while image.size(2) > min_size:
		# print(image.shape)
		image = F.interpolate(image,scale_factor=0.5,mode=mode,align_corners=False)
		# print(image)
		if normalized:
			image = F.normalize(image,dim=1,p=2)
		images.append(image)
	# if pop:
	# 	images.pop()
	images.reverse()
	return images
