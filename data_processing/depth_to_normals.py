import torch
import data_processing.optical_flow_funs as OF


def image_derivatives(image,diff_type='center', groups=3):
	c = image.size(1)
	if diff_type=='center':
		sobel_x = 0.5*torch.tensor([[0.0,0,0],[-1,0,1],[0,0,0]],device=image.device).view(1,1,3,3).repeat(c,1,1,1)
		sobel_y = 0.5*torch.tensor([[0.0,1,0],[0,0,0],[0,-1,0]],device=image.device).view(1,1,3,3).repeat(c,1,1,1)
	elif diff_type=='forward':
		sobel_x = torch.tensor([[0.0,0,0],[0,-1,1],[0,0,0]],device=image.device).view(1,1,3,3).repeat(c,1,1,1)
		sobel_y = torch.tensor([[0.0,1,0],[0,-1,0],[0,0,0]],device=image.device).view(1,1,3,3).repeat(c,1,1,1)
	
	dp_du = torch.nn.functional.conv2d(image,sobel_x,padding=1,groups=groups)
	dp_dv = torch.nn.functional.conv2d(image,sobel_y,padding=1,groups=groups)
	return dp_du, dp_dv




def point_cloud_to_normals(pc, diff_type='center'):
	#pc (b,3,m,n)
	#return (b,3,m,n)
	dp_du, dp_dv = image_derivatives(pc,diff_type=diff_type)
	normal = torch.nn.functional.normalize( torch.cross(dp_du,dp_dv,dim=1))
	return normal


def get_normals_from_depth(depth, intrinsics, depth_is_along_ray=False , diff_type='center', normalized_intrinsics=True):
	#depth (b,1,m,n)
	#intrinsics (b,3,3)
	#return (b,3,m,n), (b,3,m,n)
	dirs = OF.get_camera_pixel_directions(depth.shape, intrinsics, normalized_intrinsics=normalized_intrinsics)
	dirs = dirs.permute(0,3,1,2)
	if depth_is_along_ray:
		dirs = torch.nn.functional.normalize(dirs,dim=1)
	pc = dirs*depth
	# print(depth.shape)
	
	normal = point_cloud_to_normals(pc, diff_type=diff_type)
	return normal, depth
	

def get_normals_from_depth_list(depths, intrinsics, depth_is_along_ray=False , diff_type='center', normalized_intrinsics=True):
	#depths list[N] of (b,1,m,n)
	#intrinsics (b,3,3)
	#return list[N] of (b,3,m,n)
	normal_from_depth = []
	for depth in depths:
		nfd, _ = get_normals_from_depth(depth, intrinsics, depth_is_along_ray=depth_is_along_ray,diff_type=diff_type, normalized_intrinsics=normalized_intrinsics)
		# print(nfd.shape)
		normal_from_depth.append(nfd)
	return normal_from_depth


def get_gradient_from_depth_list(depths, diff_type='center'):
	#depths list[N] of (b,1,m,n)
	#intrinsics (b,3,3)
	#return list[N] of (b,3,m,n)
	gradient_from_depth = []
	for depth in depths:
		dp_du, dp_dv = image_derivatives(depth, diff_type=diff_type, groups=1)
		# print(nfd.shape)
		gradient = torch.cat((dp_du, dp_dv), dim=1)
		gradient_from_depth.append(gradient)
	return gradient_from_depth
