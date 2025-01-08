import torch
from data_processing.mask_triangulate import *
import data_processing.optical_flow_funs as OF
from data_processing.write_ply import *
from utils import construct_facets_from_depth_map_mask
import pyvista as pv

def save_depth_as_ply(save_name, depth, mask, intrinsics_n):
	#save_name string
	#depth (1,1,m,n)
	#mask (1,1,m,n)
	#n_intrinsics = (1,3,3)
	# print(intrinsics_n.shape)
	# intrinsics_n = OF.pixel_intrinsics_to_normalized_intrinsics(intrinsics_n.float(),(512, 612))
	# print(depth.shape)
	# print(intrinsics_n)

	dirs = OF.get_camera_pixel_directions(depth.shape,intrinsics_n).permute(0,3,1,2)
	pc = depth*dirs
	pc = pc.squeeze(0)[:,mask.squeeze()> 0.5]
	x,y,z = torch.unbind(pc,dim=0)
	
	mask = mask.squeeze().cpu().numpy() > 0.5
	faces, _, _ = mask_triangulate(mask)
	
	vert_props = [x.cpu().numpy(),-y.cpu().numpy(),-z.cpu().numpy()]


	# facets = construct_facets_from_depth_map_mask(mask)
	# vertices = pc.permute(1, 0).cpu().numpy()
	#
	# surface = pv.PolyData(vertices, facets)
	#
	# surface.save(save_name, binary=True)
	# plotter = pv.Plotter(off_screen=True)
	# plotter.add_mesh(surface, color="#BEBEBE")
	# plotter.camera_position = 'yx'
	# plotter.show(window_size=depth.shape[2:4], screenshot=save_name.replace('.ply', '.png'))
	write_ply(save_name,vert_props,prop_names=['x','y','z'],prop_types=['float32' for _ in range(0,3)],faces=faces)
