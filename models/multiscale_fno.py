import torch
import torch.nn as nn
import data_processing.optical_flow_funs as OF
from cupyx.scipy.sparse import csr_matrix
from cupyx.scipy.sparse.linalg import cg
import cupy as cp
import torch.nn.functional as F
from training_utils.loss_functions import masked_mean
from models.normal_integration_FNO import IntegationFNO
from training_utils.make_scales import make_scales
from . import model_utils
from training_utils.make_LHS import make_left_hand_side
from models.discotinuity_utils import *
from models.debug_tools import *

def sigmoid(x, k=2):
    return 1 / (1 + cp.exp(-k * x))

def normalize(attention_map, mask):
    attention_valid = attention_map[mask > 0]
    maxx = torch.max(attention_valid).item()
    minn = torch.min(attention_valid).item()
    att_scale = maxx - minn

    min_shape = torch.ones_like(attention_valid)
    min_att = min_shape * minn
    attention_valid = attention_valid - min_att
    attention_valid = torch.div(attention_valid, att_scale)
    attention_map[mask > 0] = attention_valid
    return attention_map

class AttExtractor(nn.Module):
    def __init__(self, batchNorm=True, c_in=3, other={}):
        super(AttExtractor, self).__init__()
        self.convA = model_utils.conv(batchNorm, 1, 128, k=3, stride=2, pad=1)
        self.convB = model_utils.conv(batchNorm, 128, 128, k=3, stride=1, pad=1)
        self.convD = model_utils.deconv(128, 64)

    def forward(self, xgrad):
        x3 = self.convA(xgrad)
        attention_one = self.convD(x3)
        n1, c1, h1, w1 = attention_one.data.shape
        attention_one = attention_one.view(-1)
        return attention_one, [n1, c1, h1, w1]


class AttRegressor(nn.Module):
    def __init__(self, batchNorm=True, other={}):
        super(AttRegressor, self).__init__()
        self.other = other
        self.deconv1 = model_utils.conv(batchNorm, 64, 128, k=3, stride=2, pad=1)
        self.deconv4 = model_utils.deconv(128, 64)
        self.deconv5 = self._make_output(64, 1, k=3, stride=1, pad=1)
        self.other = other

    def _make_output(self, cin, cout, k=3, stride=1, pad=1):
        return nn.Sequential(
            nn.Conv2d(cin, cout, kernel_size=k, stride=stride, padding=pad, bias=False))

    def forward(self, x, shape):
        x = x.view(shape[0], shape[1], shape[2], shape[3])
        out = self.deconv1(x)
        out = self.deconv4(out)
        attention = self.deconv5(out)
        return attention


class MultiscaleFNO(nn.Module):

    def __init__(self, use_mask=True, detach_scale=True, detach_integration=True, detach_light=True):
        super(MultiscaleFNO, self).__init__()

        self.use_mask = use_mask
        self.detach_scale = detach_scale
        self.detach_integration = detach_integration
        self.detach_light = detach_light


        self.integration_net_base = IntegationFNO()
        self.integration_net_rec = IntegationFNO()

        self.attextractor = AttExtractor()
        self.attregressor = AttRegressor()

    def print_size(self):
        nparams = 0
        nbytes = 0

        for param in self.parameters():
            nparams += param.numel()
            nbytes += param.data.element_size() * param.numel()

        print(f'Total number of model parameters: {nparams} (~{format_tensor_size(nbytes)})')

        return nparams

    def main_forward(self, normal, intrinsics, curr_depth, integration_net, mask=None):

        if self.detach_integration:
            normal_integ = normal.detach()
        else:
            normal_integ = normal

        depth = integration_net(normal_integ, intrinsics, curr_depth, mask=mask)
        attention = torch.ones_like(depth)
        m = masked_mean(depth, mask, attention).view(-1, 1, 1, 1)
        depth = depth / m.clamp(min=1e-8)

        return depth

    def forward(self, normal, n_intrinsics, mask, test_mode=False, lambda1=1e-6):
        device = normal.device
        normal_scales = make_scales(normal.float())
        mask_scales = make_scales(mask.float())
        depth_out = []
        attention_out = []

        for i, (normal_scale, mask_scale) in enumerate(zip(normal_scales, mask_scales)):

            if i == 0:

                s = normal_scale.shape
                curr_normal = normal_scale
                curr_depth = torch.ones(s[0], 1, s[2], s[3], device=device)


                curr_integ_net = self.integration_net_base


            else:
                curr_normal = normal_scale
                curr_depth = F.interpolate(depth, scale_factor=2, mode='bilinear')



                curr_integ_net = self.integration_net_rec
                if self.detach_scale:
                    curr_depth = curr_depth.detach()
                    curr_normal = curr_normal.detach()


            depth = self.main_forward(curr_normal, n_intrinsics, curr_depth, curr_integ_net, mask=mask_scale)



            u, v, denom = make_left_hand_side(curr_normal, n_intrinsics)



            attention_in = depth * denom
            h_x = attention_in.size()[2]
            w_x = attention_in.size()[3]
            dx = 2 / w_x
            dy = 2 / h_x
            r = F.pad(attention_in, (0, 1, 0, 0))[:, :, :, 1:]
            l = F.pad(attention_in, (1, 0, 0, 0))[:, :, :, :w_x]
            t = F.pad(attention_in, (0, 0, 1, 0))[:, :, :h_x, :]
            b = F.pad(attention_in, (0, 0, 0, 1))[:, :, 1:, :]
            rgrad = torch.pow((attention_in - r) / dx, 2)
            lgrad = torch.pow((attention_in - l) / dx, 2)
            tgrad = torch.pow((attention_in - t) / dy, 2)
            bgrad = torch.pow((attention_in - b) / dy, 2)

            xgrad = rgrad + lgrad + tgrad + bgrad
            att, shape = self.attextractor(xgrad)
            attentionmap = self.attregressor(att, shape)
            attention_map = normalize(attentionmap, mask_scale)

            depth_out.append(depth)
            attention_out.append(1-attention_map)

        # FNIN uses the relative weight generated by attention network. If you want use sigmoid function (FNIN-S)
        # please refer: https://github.com/xucao-42/bilateral_normal_integration/tree/main
        if test_mode:
            curr_depth = cp.asarray(depth.squeeze().squeeze().cpu().numpy())
            curr_normal = cp.asarray(curr_normal.squeeze().cpu().numpy())
            curr_mask = cp.asarray(mask_scale.squeeze().squeeze().cpu().numpy())
            curr_mask = curr_mask > 0
            curr_attention = cp.asarray(attention_map.squeeze().squeeze().cpu().numpy())

            num_normals = cp.sum(curr_mask).item()

            nx = curr_normal[1, curr_mask]
            ny = - curr_normal[0, curr_mask]
            nz = - curr_normal[2, curr_mask]

            if n_intrinsics is not None:  # perspective
                img_height, img_width = curr_mask.shape
                K = OF.normalized_intrinsics_to_pixel_intrinsics(n_intrinsics, curr_mask.shape)
                K = cp.asarray(K.squeeze().cpu().numpy())

                yy, xx = cp.meshgrid(cp.arange(img_width), cp.arange(img_height))
                xx = cp.flip(xx, axis=0)

                cx = K[1, 2]
                cy = K[0, 2]
                fx = K[1, 1]
                fy = K[0, 0]

                uu = xx[curr_mask] - cx
                vv = yy[curr_mask] - cy

                nz_u = uu * nx + vv * ny + fx * nz
                nz_v = uu * nx + vv * ny + fy * nz
                del xx, yy, uu, vv
            else:  # orthographic
                nz_u = nz.copy()
                nz_v = nz.copy()


            A3, A4, A1, A2 = generate_dx_dy(curr_mask, nz_horizontal=nz_v, nz_vertical=nz_u, step_size=1)

            pixel_idx = cp.zeros_like(curr_mask, dtype=int)
            pixel_idx[curr_mask] = cp.arange(num_normals)
            pixel_idx_flat = cp.arange(num_normals)
            pixel_idx_flat_indptr = cp.arange(num_normals + 1)

            has_left_mask = cp.logical_and(move_right(curr_mask), curr_mask)
            has_left_mask_left = move_left(has_left_mask)
            has_right_mask = cp.logical_and(move_left(curr_mask), curr_mask)
            has_right_mask_right = move_right(has_right_mask)
            has_bottom_mask = cp.logical_and(move_top(curr_mask), curr_mask)
            has_bottom_mask_bottom = move_bottom(has_bottom_mask)
            has_top_mask = cp.logical_and(move_bottom(curr_mask), curr_mask)
            has_top_mask_top = move_top(has_top_mask)

            has_left_mask_flat = has_left_mask[curr_mask]
            has_right_mask_flat = has_right_mask[curr_mask]
            has_bottom_mask_flat = has_bottom_mask[curr_mask]
            has_top_mask_flat = has_top_mask[curr_mask]

            has_left_mask_left_flat = has_left_mask_left[curr_mask]
            has_right_mask_right_flat = has_right_mask_right[curr_mask]
            has_bottom_mask_bottom_flat = has_bottom_mask_bottom[curr_mask]
            has_top_mask_top_flat = has_top_mask_top[curr_mask]

            nz_left_square = nz_v[has_left_mask_flat] ** 2
            nz_right_square = nz_v[has_right_mask_flat] ** 2
            nz_top_square = nz_u[has_top_mask_flat] ** 2
            nz_bottom_square = nz_u[has_bottom_mask_flat] ** 2

            pixel_idx_left_center = pixel_idx[has_left_mask]
            pixel_idx_right_right = pixel_idx[has_right_mask_right]
            pixel_idx_top_center = pixel_idx[has_top_mask]
            pixel_idx_bottom_bottom = pixel_idx[has_bottom_mask_bottom]

            pixel_idx_left_left_indptr = cp.concatenate([cp.array([0]), cp.cumsum(has_left_mask_left_flat)])
            pixel_idx_right_center_indptr = cp.concatenate([cp.array([0]), cp.cumsum(has_right_mask_flat)])
            pixel_idx_top_top_indptr = cp.concatenate([cp.array([0]), cp.cumsum(has_top_mask_top_flat)])
            pixel_idx_bottom_center_indptr = cp.concatenate([cp.array([0]), cp.cumsum(has_bottom_mask_flat)])

            # Construct the linear system
            depth_mask_flat = curr_mask[curr_mask].astype(bool)  # shape: (num_normals,)
            z_prior = cp.log(curr_depth)[curr_mask] if n_intrinsics is not None else curr_depth[
                curr_mask]  # shape: (num_normals,)
            z_prior[~depth_mask_flat] = 0

            z = z_prior


            # the weight here may be different from paper, but it is consistent with the loss in criterion.py
            W = curr_attention / 2
            h_x = W.shape[0]
            w_x = W.shape[1]

            wr = cp.pad(W, ((0,0),(0,1)),'constant',constant_values=1)[:, 1:]
            wl = cp.pad(W, ((0,0),(1,0)),'constant',constant_values=1)[:, :w_x]
            wt = cp.pad(W, ((0,1),(0,0)),'constant',constant_values=1)[:h_x, :]
            wb = cp.pad(W, ((1,0),(0,0)),'constant',constant_values=1)[1:, :]


            wr = wr[curr_mask]
            wl = wl[curr_mask]
            wt = wt[curr_mask]
            wb = wb[curr_mask]


            data_term_top = wt[has_top_mask_flat] * nz_top_square
            data_term_bottom = wb[has_bottom_mask_flat] * nz_bottom_square
            data_term_left = wl[has_left_mask_flat] * nz_left_square
            data_term_right = wr[has_right_mask_flat] * nz_right_square

            diagonal_data_term = cp.zeros(num_normals)
            diagonal_data_term[has_left_mask_flat] += data_term_left
            diagonal_data_term[has_left_mask_left_flat] += data_term_left
            diagonal_data_term[has_right_mask_flat] += data_term_right
            diagonal_data_term[has_right_mask_right_flat] += data_term_right
            diagonal_data_term[has_top_mask_flat] += data_term_top
            diagonal_data_term[has_top_mask_top_flat] += data_term_top
            diagonal_data_term[has_bottom_mask_flat] += data_term_bottom
            diagonal_data_term[has_bottom_mask_bottom_flat] += data_term_bottom

            diagonal_data_term[depth_mask_flat] += lambda1

            A_mat_d = csr_matrix((diagonal_data_term, pixel_idx_flat, pixel_idx_flat_indptr),
                                 shape=(num_normals, num_normals))

            A_mat_left_odu = csr_matrix((-data_term_left, pixel_idx_left_center, pixel_idx_left_left_indptr),
                                        shape=(num_normals, num_normals))
            A_mat_right_odu = csr_matrix((-data_term_right, pixel_idx_right_right, pixel_idx_right_center_indptr),
                                         shape=(num_normals, num_normals))
            A_mat_top_odu = csr_matrix((-data_term_top, pixel_idx_top_center, pixel_idx_top_top_indptr),
                                       shape=(num_normals, num_normals))
            A_mat_bottom_odu = csr_matrix((-data_term_bottom, pixel_idx_bottom_bottom, pixel_idx_bottom_center_indptr),
                                          shape=(num_normals, num_normals))

            A_mat_odu = A_mat_top_odu + A_mat_bottom_odu + A_mat_right_odu + A_mat_left_odu
            A_mat = A_mat_d + A_mat_odu + A_mat_odu.T


            D = csr_matrix((1 / cp.clip(diagonal_data_term, 1e-5, None), pixel_idx_flat, pixel_idx_flat_indptr),
                           shape=(num_normals, num_normals))
            b_vec = A1.T @ (wt * (-nx)) \
                    + A2.T @ (wb * (-nx)) \
                    + A3.T @ (wr * (-ny)) \
                    + A4.T @ (wl * (-ny))

            b_vec += lambda1 * z_prior

            z, _ = cg(A_mat, b_vec, x0=z, M=D, maxiter=5000, tol=1e-3)

            curr_depth[curr_mask] = cp.exp(z)
            curr_depth = cp.asnumpy(curr_depth)
            curr_depth = torch.from_numpy(curr_depth).unsqueeze(0).unsqueeze(0).cuda()


            attention = torch.ones_like(curr_depth)
            m = masked_mean(curr_depth, mask_scale, attention).view(-1, 1, 1, 1)
            curr_depth = curr_depth / m.clamp(min=1e-8)
            depth_out[-1] = curr_depth




        outputs = {'depth_scales': depth_out, 'attention_scales': attention_out}

        return outputs



