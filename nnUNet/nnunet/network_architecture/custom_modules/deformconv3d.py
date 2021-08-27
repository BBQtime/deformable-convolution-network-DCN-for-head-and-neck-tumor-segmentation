from torch.autograd import Variable, Function
import torch
from torch import nn
import numpy as np

class DeformConv3D(nn.Module):
    """
    refrence: https://github.com/KrakenLeaf/deform_conv_pytorch/blob/master/deform_conv_3d.py
              https://github.com/kondratevakate/3d-deformable-convolutions
    """
    def __init__(self, inc, outc=[], kernel_size=3, padding=1, bias=None):
        super(DeformConv3D, self).__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        #self.zero_padding = nn.functional.pad(padding)
        self.conv_kernel = nn.Conv3d(inc, outc, kernel_size=kernel_size, stride=kernel_size, bias=bias)

    def forward(self, x, offset):
        dtype = offset.data.type()
        ks = self.kernel_size

        # Out_channels = 3 * kernel_size_x * kernel_size_y * kernel_size_z
        M = offset.size(1)
        N = M // 3 # Number of channels

        if self.padding != 0:
            # For simplicity we pad from both sides in all 3 dimensions
            padding_use = (self.padding, self.padding, self.padding, self.padding ,self.padding, self.padding)
            x = nn.functional.pad(x, padding_use, "constant", 0)

        # Get input dimensions
        b, c, h, w, d = x.size()
        shape = (h, w, d)

        # interpolation points p (Eq. 2)
        # ------------------------------
        # (b, 3N, h, w, d)
        p = self._get_p(offset, dtype)

        # (b, h, w, d, 3N)
        p = p.contiguous().permute(0, 2, 3, 4, 1) # p = p_0 + p_n + offset

        # Use grid_sample to interpolate
        # ------------------------------
        for ii in range(N):
            # Normalize flow field to rake values in the range [-1, 1]
            flow = p[..., [t for t in range(ii, M, N)]]
            for jj in range(3):
                flow[..., jj] = 2 * flow[..., jj] / (shape[jj] - 1) - 1

            # Push through the spatial transformer
            tmp = nn.functional.grid_sample(input=x, grid=flow.float(), mode='bilinear', padding_mode='border').contiguous()
            tmp = tmp.unsqueeze(dim=-1)

            # Aggregate
            if ii == 0:
                xt = tmp
            else:
                xt = torch.cat((xt, tmp), dim=-1)

        # For simplicity, ks is a scalar, implying kernel has same dimensions in all directions
        x_offset = self._reshape_x_offset(xt, ks)
        out = self.conv_kernel(x_offset)

        return out

    def _get_p_n(self, N, dtype):
        p_n_x, p_n_y, p_n_z = np.meshgrid(range(-(self.kernel_size - 1) // 2, (self.kernel_size - 1) // 2 + 1),
                                          range(-(self.kernel_size - 1) // 2, (self.kernel_size - 1) // 2 + 1),
                                          range(-(self.kernel_size - 1) // 2, (self.kernel_size - 1) // 2 + 1),
                                          indexing='ij')
        # (2N, 1)
        p_n = np.concatenate((p_n_x.flatten(), p_n_y.flatten(), p_n_z.flatten()))
        p_n = np.reshape(p_n, (1, 3*N, 1, 1, 1))
        p_n = Variable(torch.from_numpy(p_n).type(dtype), requires_grad=False)

        return p_n

    @staticmethod
    def _get_p_0(h, w, d, N, dtype):
        #p_0_x, p_0_y, p_0_z = np.meshgrid(range(1, h + 1), range(1, w + 1), range(1, d + 1), indexing='ij') # 1,...,N
        p_0_x, p_0_y, p_0_z = np.meshgrid(range(0, h), range(0, w), range(0, d), indexing='ij') # 0,...N-1
        p_0_x = p_0_x.flatten().reshape(1, 1, h, w, d).repeat(N, axis=1)
        p_0_y = p_0_y.flatten().reshape(1, 1, h, w, d).repeat(N, axis=1)
        p_0_z = p_0_z.flatten().reshape(1, 1, h, w, d).repeat(N, axis=1)
        p_0 = np.concatenate((p_0_x, p_0_y, p_0_z), axis=1)
        p_0 = Variable(torch.from_numpy(p_0).type(dtype), requires_grad=False)

        return p_0

    def _get_p(self, offset, dtype):
        N, h, w, d = offset.size(1)//3, offset.size(2), offset.size(3), offset.size(4)

        # (1, 3N, 1, 1, 1)
        p_n = self._get_p_n(N, dtype)
        # (1, 3N, h, w, d)
        p_0 = self._get_p_0(h, w, d, N, dtype)
        p = p_0 + p_n + offset
        return p

    def _get_x_q(self, x, q, N):
        b, h, w, d, _ = q.size()
        ny = x.size(3) # Padded dimension y
        nz = x.size(4)  # Padded dimension z
        c = x.size(1) # Number of channels in input x
        # (b, c, h*w)
        x = x.contiguous().view(b, c, -1)

        # (b, h, w, d, N)
        offset_x = q[..., :N]
        offset_y = q[..., N:2*N]
        offset_z = q[..., 2*N:]
        # Convert subscripts to linear indices (i.e. Matlab's sub2ind)
        index = offset_x * ny * nz + offset_y * nz + offset_z
        # (b, c, h*w*d*N)
        index = index.contiguous().unsqueeze(dim=1).expand(-1, c, -1, -1, -1, -1).contiguous().view(b, c, -1)

        x_offset = x.gather(dim=-1, index=index).contiguous().view(b, c, h, w, d, N)

        return x_offset

    @staticmethod
    def _reshape_x_offset(x_offset, ks):
        '''
        This function arranges the interpolated x values in consecutive 3d blocks of size
        kernel_size x kernel_size x kernel_size. Since the Conv3d stride is equal to kernel_size, the convolution
        will happen only for the offset cubes and output the results in the proper locations
        Note: We assume kernel size is the same for all dimensions (cube)
        '''
        b, c, h, w, d, N = x_offset.size()
        x_offset = torch.cat([x_offset[..., s:s + ks*ks].contiguous().view(b, c, h, w, d*ks*ks) for s in range(0, N, ks*ks)], dim=-1)
        N = x_offset.size(4)
        x_offset = torch.cat([x_offset[..., s:s + d*ks*ks].contiguous().view(b, c, h, w*ks, d*ks) for s in range(0, N, d*ks*ks)], dim=-1)
        x_offset = x_offset.contiguous().view(b, c, h*ks, w*ks, d*ks)

        return x_offset

# Alternative realization
# ----------------------------
# This realization directly extends the 2D vewrsion. However, for large dimensions this approach is very memory consuming, although
# faster than the previous approach
class DeformConv3D_alternative(nn.Module):
    def __init__(self, inc, outc=[], kernel_size=3, padding=1, bias=None):
        super(DeformConv3D_alternative, self).__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        #self.zero_padding = nn.functional.pad(padding)
        self.conv_kernel = nn.Conv3d(inc, outc, kernel_size=kernel_size, stride=kernel_size, bias=bias)

    def forward(self, x, offset):
        dtype = offset.data.type()
        ks = self.kernel_size

        # Out_channels = 3 * kernel_size_x * kernel_size_y * kernel_size_z
        N = offset.size(1) // 3 # Number of channels

        if self.padding != 0:
            # For simplicity we pad from both sides in all 3 dimensions
            padding_use = (self.padding, self.padding, self.padding, self.padding ,self.padding, self.padding)
            x = nn.functional.pad(x, padding_use, "constant", 0)

        # interpolation points p (Eq. 2)
        # ------------------------------
        # (b, 3N, h, w, d)
        p = self._get_p(offset, dtype)

        # (b, h, w, d, 3N)
        p = p.contiguous().permute(0, 2, 3, 4, 1) # p = p_0 + p_n + offset

        # Regular grid points q (Eq. 3)
        # -----------------------------
        q_lt = Variable(p.data, requires_grad=False).floor()
        q_rb = q_lt + 1

        # Enumerate all integral locations in the feature map x
        # Clamp values between 0 and size of input, per each direction XYZ
        # OS: lt - Left/Top, rt - Right/Top, lb - Left/Bottom, rb - Right/Bottom?
        q_000 = torch.cat([torch.clamp(q_lt[..., :N], 0, x.size(2) - 1), # x.size() = b X c X h X w X d
                          torch.clamp(q_lt[..., N:2*N], 0, x.size(3) - 1),
                          torch.clamp(q_lt[..., 2*N:], 0, x.size(4) - 1)
                          ], dim=-1).long()
        q_111 = torch.cat([torch.clamp(q_rb[..., :N], 0, x.size(2) - 1),
                          torch.clamp(q_rb[..., N:2*N], 0, x.size(3) - 1),
                          torch.clamp(q_rb[..., 2*N:], 0, x.size(4) - 1)
                          ], dim=-1).long()
        q_001 = torch.cat([q_000[..., :N], q_000[..., N:2 * N], q_111[..., 2 * N:]], dim=-1)
        q_010 = torch.cat([q_000[..., :N], q_111[..., N:2 * N], q_000[..., 2 * N:]], dim=-1)
        q_011 = torch.cat([q_000[..., :N], q_111[..., N:2 * N], q_111[..., 2 * N:]], dim=-1)
        q_100 = torch.cat([q_111[..., :N], q_000[..., N:2 * N], q_000[..., 2 * N:]], dim=-1)
        q_101 = torch.cat([q_111[..., :N], q_000[..., N:2 * N], q_111[..., 2 * N:]], dim=-1)
        q_110 = torch.cat([q_111[..., :N], q_111[..., N:2 * N], q_000[..., 2 * N:]], dim=-1)

        # (b, h, w, d, N)
        mask = torch.cat([p[..., :N].lt(self.padding) + p[..., :N].gt(x.size(2) - 1 - self.padding),
                          p[..., N:2*N].lt(self.padding) + p[..., N:2*N].gt(x.size(3) - 1 - self.padding),
                          p[..., 2*N:].lt(self.padding) + p[..., 2*N:].gt(x.size(4) - 1 - self.padding)
                          ], dim=-1).type_as(p)
        mask = mask.detach() # Detach from computational graph
        floor_p = p - (p - torch.floor(p))
        p = p*(1-mask) + floor_p*mask
        p = torch.cat([torch.clamp(p[..., :N], 0, x.size(2) - 1),
                       torch.clamp(p[..., N:2*N], 0, x.size(3) - 1),
                       torch.clamp(p[..., 2*N:], 0, x.size(4) - 1)
                       ], dim=-1)

        # Interpolation kernel - x(q) (Eq. 4)
        # -----------------------------------
        # bilinear kernel (b, h, w, d, N)
        g_000 = (1 + (q_000[..., :N].type_as(p) - p[..., :N])) * (1 + (q_000[..., N:2 * N].type_as(p) - p[..., N:2 * N])) * (1 + (q_000[..., 2 * N:].type_as(p) - p[..., 2 * N:]))
        g_111 = (1 - (q_111[..., :N].type_as(p) - p[..., :N])) * (1 - (q_111[..., N:2 * N].type_as(p) - p[..., N:2 * N])) * (1 - (q_111[..., 2 * N:].type_as(p) - p[..., 2 * N:]))
        g_001 = (1 + (q_000[..., :N].type_as(p) - p[..., :N])) * (1 + (q_000[..., N:2 * N].type_as(p) - p[..., N:2 * N])) * (1 - (q_111[..., 2 * N:].type_as(p) - p[..., 2 * N:]))
        g_010 = (1 + (q_000[..., :N].type_as(p) - p[..., :N])) * (1 - (q_111[..., N:2 * N].type_as(p) - p[..., N:2 * N])) * (1 + (q_000[..., 2 * N:].type_as(p) - p[..., 2 * N:]))
        g_011 = (1 + (q_000[..., :N].type_as(p) - p[..., :N])) * (1 - (q_111[..., N:2 * N].type_as(p) - p[..., N:2 * N])) * (1 - (q_111[..., 2 * N:].type_as(p) - p[..., 2 * N:]))
        g_100 = (1 - (q_111[..., :N].type_as(p) - p[..., :N])) * (1 + (q_000[..., N:2 * N].type_as(p) - p[..., N:2 * N])) * (1 + (q_000[..., 2 * N:].type_as(p) - p[..., 2 * N:]))
        g_101 = (1 - (q_111[..., :N].type_as(p) - p[..., :N])) * (1 + (q_000[..., N:2 * N].type_as(p) - p[..., N:2 * N])) * (1 - (q_111[..., 2 * N:].type_as(p) - p[..., 2 * N:]))
        g_110 = (1 - (q_111[..., :N].type_as(p) - p[..., :N])) * (1 - (q_111[..., N:2 * N].type_as(p) - p[..., N:2 * N])) * (1 + (q_000[..., 2 * N:].type_as(p) - p[..., 2 * N:]))

        # Interpolation - x(q) (Eq. 3)
        # ----------------------------
        # (b, c, h, w, d, N)
        x_q_000 = self._get_x_q(x, q_000, N)
        x_q_111 = self._get_x_q(x, q_111, N)
        x_q_001 = self._get_x_q(x, q_001, N)
        x_q_010 = self._get_x_q(x, q_010, N)
        x_q_011 = self._get_x_q(x, q_011, N)
        x_q_100 = self._get_x_q(x, q_100, N)
        x_q_101 = self._get_x_q(x, q_101, N)
        x_q_110 = self._get_x_q(x, q_110, N)

        # (b, c, h, w, d, N)
        x_offset = g_000.unsqueeze(dim=1) * x_q_000 + \
                   g_111.unsqueeze(dim=1) * x_q_111 + \
                   g_001.unsqueeze(dim=1) * x_q_001 + \
                   g_010.unsqueeze(dim=1) * x_q_010 + \
                   g_011.unsqueeze(dim=1) * x_q_011 + \
                   g_100.unsqueeze(dim=1) * x_q_100 + \
                   g_101.unsqueeze(dim=1) * x_q_101 + \
                   g_110.unsqueeze(dim=1) * x_q_110

        # For simplicity, ks is a scalar, implying kernel has same dimensions in all directions
        x_offset = self._reshape_x_offset(x_offset, ks)
        out = self.conv_kernel(x_offset)

        return out

    def _get_p_n(self, N, dtype):
        p_n_x, p_n_y, p_n_z = np.meshgrid(range(-(self.kernel_size - 1) // 2, (self.kernel_size - 1) // 2 + 1),
                                          range(-(self.kernel_size - 1) // 2, (self.kernel_size - 1) // 2 + 1),
                                          range(-(self.kernel_size - 1) // 2, (self.kernel_size - 1) // 2 + 1),
                                          indexing='ij')
        # (2N, 1)
        p_n = np.concatenate((p_n_x.flatten(), p_n_y.flatten(), p_n_z.flatten()))
        p_n = np.reshape(p_n, (1, 3*N, 1, 1, 1))
        p_n = Variable(torch.from_numpy(p_n).type(dtype), requires_grad=False)

        return p_n

    @staticmethod
    def _get_p_0(h, w, d, N, dtype):
        p_0_x, p_0_y, p_0_z = np.meshgrid(range(1, h+1), range(1, w+1), range(1, d+1), indexing='ij')
        p_0_x = p_0_x.flatten().reshape(1, 1, h, w, d).repeat(N, axis=1)
        p_0_y = p_0_y.flatten().reshape(1, 1, h, w, d).repeat(N, axis=1)
        p_0_z = p_0_z.flatten().reshape(1, 1, h, w, d).repeat(N, axis=1)
        p_0 = np.concatenate((p_0_x, p_0_y, p_0_z), axis=1)
        p_0 = Variable(torch.from_numpy(p_0).type(dtype), requires_grad=False)

        return p_0

    def _get_p(self, offset, dtype):
        N, h, w, d = offset.size(1)//3, offset.size(2), offset.size(3), offset.size(4)

        # (1, 3N, 1, 1, 1)
        p_n = self._get_p_n(N, dtype)
        # (1, 3N, h, w, d)
        p_0 = self._get_p_0(h, w, d, N, dtype)
        p = p_0 + p_n + offset
        return p

    def _get_x_q(self, x, q, N):
        b, h, w, d, _ = q.size()
        ny = x.size(3) # Padded dimension y
        nz = x.size(4)  # Padded dimension z
        c = x.size(1) # Number of channels in input x
        # (b, c, h*w)
        x = x.contiguous().view(b, c, -1)

        # (b, h, w, d, N)
        offset_x = q[..., :N]
        offset_y = q[..., N:2*N]
        offset_z = q[..., 2*N:]
        # Convert subscripts to linear indices (i.e. Matlab's sub2ind)
        index = offset_x * ny * nz + offset_y * nz + offset_z
        # (b, c, h*w*d*N)
        index = index.contiguous().unsqueeze(dim=1).expand(-1, c, -1, -1, -1, -1).contiguous().view(b, c, -1)

        x_offset = x.gather(dim=-1, index=index).contiguous().view(b, c, h, w, d, N)

        return x_offset

    @staticmethod
    def _reshape_x_offset(x_offset, ks):
        '''
        This function arranges the interpolated x values in consecutive 3d blocks of size
        kernel_size x kernel_size x kernel_size. Since the Conv3d stride is equal to kernel_size, the convolution
        will happen only for the offset cubes and output the results in the proper locations
        Note: We assume kernel size is the same for all dimensions (cube)
        '''
        b, c, h, w, d, N = x_offset.size()
        x_offset = torch.cat([x_offset[..., s:s + ks*ks].contiguous().view(b, c, h, w, d*ks*ks) for s in range(0, N, ks*ks)], dim=-1)
        N = x_offset.size(4)
        x_offset = torch.cat([x_offset[..., s:s + d*ks*ks].contiguous().view(b, c, h, w*ks, d*ks) for s in range(0, N, d*ks*ks)], dim=-1)
        x_offset = x_offset.contiguous().view(b, c, h*ks, w*ks, d*ks)

        return x_offset


class DeformBasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1):
        super(DeformBasicBlock, self).__init__()
        self.output_channels = planes
        self.in_planes = inplanes
        self.offsets = conv3x3x3(inplanes, 81, stride) # why 81? out_channel: kenerl_size * xyz 3 directions 

        self.conv1 = DeformConv3D_alternative(inplanes, planes*2) # ! inplanes, planes
        #self.conv1 = DeformConv3D_alternative(inplanes, planes)
        self.bn1 = nn.BatchNorm3d(planes*2) # ! (planes)
        self.lrelu = nn.LeakyReLU(inplace=True)
        self.conv2 = conv3x3x3(planes*2, planes) # ! (planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.stride = stride

    def forward(self, x):
        #residual = x
        #print("!!!in_planes:", self.in_planes)
    
        # deformable convolution offsets
        offsets = self.offsets(x)
        out = self.conv1(x, offsets)
        out = self.lrelu(out)
        out = self.bn1(out)
        out = self.conv2(out)
        out = self.bn2(out)

        #out += residual
        out = self.lrelu(out)

        return out
    
    
class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class Identity(nn.Module):
    def __init__(self,):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x


class Reshape(nn.Module):
    def __init__(self, shape):
        nn.Module.__init__(self)
        self.shape = shape
    def forward(self, input):
        return input.view((-1,) + self.shape)


def conv3x3x3(in_planes, out_planes, stride=1):
    # 3x3x3 convolution with padding
    return nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

def conv1x1x1(in_planes, out_planes, stride=1):
    # 3x3x3 convolution with padding
    return nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=stride, padding=1, bias=False)

class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.LeakyReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)

        return out