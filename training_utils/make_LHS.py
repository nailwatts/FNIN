import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt
import data_processing.optical_flow_funs as OF


def image_derivativesLHS(image,side=1):
	# print(image.shape)
	c = image.size(1)
	sobel_x = 0.5*torch.tensor([[0.0,0,0],[-1,0,1],[0,0,0.0]],device=image.device).view(1,1,3,3).repeat(c,1,1,1)
	sobel_y = 0.5*torch.tensor([[0,-1,0],[0,0,0],[0,1,0.0]],device=image.device).view(1,1,3,3).repeat(c,1,1,1)
	dp_du = torch.nn.functional.conv2d(image,sobel_x,padding=1,groups=c)
	dp_dv = torch.nn.functional.conv2d(image,sobel_y,padding=1,groups=c)

	return dp_du, dp_dv


# def image_derivativesLHS(image, sobel_x, sobel_y):
# 	# print(image.shape)
# 	c = image.size(1)
# 	# sobel_x = nn.Parameter(0.5*torch.tensor([[0.0,0,0],[-1,0,1],[0,0,0.0]],device=image.device).view(1,1,3,3).repeat(c,1,1,1), requires_grad=True)
# 	# sobel_y = nn.Parameter(0.5*torch.tensor([[0,-1,0],[0,0,0],[0,1,0.0]],device=image.device).view(1,1,3,3).repeat(c,1,1,1), requires_grad=True)
#
# 	dp_du = torch.nn.functional.conv2d(image,sobel_x,padding=1,groups=c)
# 	dp_dv = torch.nn.functional.conv2d(image,sobel_y,padding=1,groups=c)
#
# 	return dp_du, dp_dv


# def image_derivativesLHS(image, sobel_x, sobel_y):
# 	# print(image.shape)
# 	c = image.size(1)
# 	# sobel_x = nn.Parameter(0.5*torch.tensor([[0.0,0,0],[-1,0,1],[0,0,0.0]],device=image.device).view(1,1,3,3).repeat(c,1,1,1), requires_grad=True)
# 	# sobel_y = nn.Parameter(0.5*torch.tensor([[0,-1,0],[0,0,0],[0,1,0.0]],device=image.device).view(1,1,3,3).repeat(c,1,1,1), requires_grad=True)
#
# 	dp_du = torch.nn.functional.conv2d(image,sobel_x,padding=1,groups=c)
# 	dp_dv = torch.nn.functional.conv2d(image,sobel_y,padding=1,groups=c)
#
# 	return dp_du, dp_dv
	


def make_left_hand_side(normals,n_intrinsics):
	#(b,3,m,n), (b,3,3)
	#return (b,1,m,n)
	# print(normals.shape)
	if n_intrinsics.shape[-1] > 1:
		grid = OF.get_coordinate_grid(normals.shape[2:4],device=normals.device)
		n_grid = OF.pixel_coords_to_normalized_coords(grid,normals.shape[2:4])

		#(b,m,n)
		x = n_grid[:,:,:,0]
		y = n_grid[:,:,:,1]


		f1 = n_intrinsics[:,0,0].view(-1,1,1)
		f2 = n_intrinsics[:,1,1].view(-1,1,1)
		a = n_intrinsics[:,0,2].view(-1,1,1)
		b = n_intrinsics[:,1,2].view(-1,1,1)

		#(b,m,n)
		n1,n2,n3 = torch.unbind(normals, dim=1)

		f1_times_n2 = f1*n2
		f2_times_n1 = f2*n1



		denom = f2_times_n1*(x-a) + f1_times_n2*(y-b) + f1*f2*n3
		nz = denom

		# denom1 = n1 * (x - a) + n2 * (y - b) + f1 * n3
		denom[(denom < 1e-8) & (denom > -1e-8)] = 1

		# denom[denom<-10] = -10
		# print(denom.shape)
		# denom2 = n1 * (x - a) + n2 * (y - b) + f2 * n3
		# denom2[(denom2 < 1e-8) & (denom2 > -1e-8)] = 1

		# u_x = n1 / denom1
		# u_y = n2 / denom2

		u_x = -f2_times_n1/denom
		u_y = -f1_times_n2/denom
	else:
		# print(n_intrinsics)
		nx = normals[:, 0, :, :]
		ny = normals[:, 1, :, :]
		nz = normals[:, 2, :, :]
		denom = nz
		denom[(denom < 1e-8) & (denom > -1e-8)] = 1
		u_x = -nx / denom
		u_y = -ny / denom
		# print(normals.shape)

	# print(u_x)
	
	return u_x.unsqueeze(1), u_y.unsqueeze(1), nz.unsqueeze(1)

