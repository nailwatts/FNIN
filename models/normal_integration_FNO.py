import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np
from training_utils.make_LHS import *

from training_utils.make_LHS import make_left_hand_side
from models.neuralop.models import TFNO


class IntegationFNO(nn.Module):
	def __init__(self, use_mask=True, use_tanh=True, init_feature_nc=32, use_f=True):
		super(IntegationFNO, self).__init__()
		if use_mask:
			self.net = TFNO(n_modes=(16, 16), in_channels=4, hidden_channels=32, projection_channels=64, factorization='tucker', rank=0.42)

		# self.sobel_x = nn.Parameter(
		# 	0.5 * torch.tensor([[0.0, 0, 0], [-1, 0, 1], [0, 0, 0.0]],).view(1, 1, 3, 3).repeat(1,1,1,1),
		# 	requires_grad=True)
		#
		# self.sobel_y = nn.Parameter(
		# 	0.5 * torch.tensor([[0,-1,0],[0,0,0],[0,1,0.0]], ).view(1, 1, 3, 3).repeat(1, 1, 1, 1),
		# 	requires_grad=True)

		
	def del_op(self, f):
		delta_x_i = 2/f.size(3)
		delta_y_i = 2/f.size(2)
		f_x, f_y = image_derivativesLHS(f)
		f_x = f_x/delta_x_i
		f_y = f_y/delta_y_i
		return f_x, f_y

	def get_grid(self, shape, device):
		grid = OF.get_coordinate_grid(shape, device=device)
		n_grid = OF.pixel_coords_to_normalized_coords(grid, shape)
		return n_grid.permute(0, 3, 1, 2)

	def main_forward(self, u, v, curr_log_depth, mask=None, denom=None):

		# print(torch.isnan(curr_log_depth).sum())
	
		f_x, f_y = self.del_op(curr_log_depth)
		# print(torch.isnan(f_x).sum())
		# print(curr_log_depth.shape)
		# print(f_x.shape)
			
		u_bar = u - f_x# * denom
		v_bar = v - f_y# * denom
		# u_bar = a1 * f_x - b1
		# v_bar = a2 * f_y - b2

		# dx = 2/u_x.size(3)
		# dy = 2/u_y.size(2)
		grid = self.get_grid(u_bar.shape[2:4], u_bar.device)
		grid = grid.expand(u_bar.shape[0], -1, -1, -1)
		# print(grid.shape)
		# print(u_x.shape)
		# t = u_x * dx
		# print(t[:, :, 32, :])
		if mask is not None:
			# net_in = torch.cat((u_x * dx, u_y * dy, mask), dim=1)
			net_in = torch.cat((u_bar * mask, v_bar * mask, grid), dim=1)
			# net_in = torch.cat((u_bar, v_bar, mask, grid), dim=1)


		# print(torch.isnan(net_in).sum())

		g = self.net(net_in)

		# print(torch.isnan(net_out).sum())

		g = F.tanh(g)

		# print(torch.isnan(u_bar).sum())

		# print(torch.isinf(g).sum())

			
		log_depth = curr_log_depth + g #/ denom
			
		return log_depth


	def forward(self, normal, intrinsics, curr_depth, mask=None, gt_depth=None):
		u, v, denom = make_left_hand_side(normal,intrinsics) #adjust for intrinsics
		# print(u[:, :, 32, :])
		# print(torch.isnan(u).sum())
		# print(curr_depth.shape)

		if intrinsics is not None:
			curr_depth = torch.log(curr_depth.clamp(min=1e-8))
		# print(curr_log_depth.shape)
		# print(torch.sum(denom<0))
		# print(torch.isnan(curr_depth).sum())

		depth = self.main_forward(u, v, curr_depth, mask=mask, denom=denom)
		# print(torch.isnan(depth).sum())

		if intrinsics is not None:
			depth = torch.exp(depth)



		return depth
			
			
			
		
		
		
		
		
		

