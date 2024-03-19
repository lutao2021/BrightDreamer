import torch
import torch.nn as nn
from .network_crossattn_generative import TriplaneGenerator as CrossAttnTriplaneGenerator
from .point_transformer import PointTransformer

class Generator(nn.Module):
    def __init__(self, input_dim=3, z_dim=1024, context_dim=4096, hidden_dim=256, output_dim=1, device=None, opt=None, bound=1):
        super().__init__()
        self.device = device
        self.input_dim = input_dim
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.context_dim = context_dim
        self.opt = opt
        self.bound = bound
        self.free_distance = self.opt.free_distance

        self.point_net = PointTransformer(in_channels=3, n_heads=1, d_head=64//1, depth=2, context_dim=self.context_dim, disable_self_attn=True, opt=self.opt)

        self.triplane_generator_XY = CrossAttnTriplaneGenerator(use_fp16=self.opt.fp16, opt=self.opt, out_channels=32, context_dim=self.context_dim)
        self.triplane_generator_XZ = CrossAttnTriplaneGenerator(use_fp16=self.opt.fp16, opt=self.opt, out_channels=32, context_dim=self.context_dim)
        self.triplane_generator_YZ = CrossAttnTriplaneGenerator(use_fp16=self.opt.fp16, opt=self.opt, out_channels=32, context_dim=self.context_dim)

        if self.opt.xyzres:
            self.shape_transform = nn.Sequential(nn.Linear(32 + 3, 64), nn.Softplus(), nn.Linear(64, 8))
            self.shape_transform[-1].bias.data[0] = -2.0
            self.color_transform = nn.Sequential(nn.Linear(32 + 3, 64), nn.Softplus(), nn.Linear(64, 3))
        else:
            self.shape_transform = nn.Sequential(nn.Linear(32, 64), nn.Softplus(), nn.Linear(64, 8))
            self.shape_transform[-1].bias.data[0] = -2.0
            self.color_transform = nn.Sequential(nn.Linear(32, 64), nn.Softplus(), nn.Linear(64, 3))

        self.register_buffer('plane_axes', torch.tensor([[[1, 0, 0],
                                                          [0, 1, 0],
                                                          [0, 0, 1]],
                                                         [[1, 0, 0],
                                                          [0, 0, 1],
                                                          [0, 1, 0]],
                                                         [[0, 0, 1],
                                                          [1, 0, 0],
                                                          [0, 1, 0]]], dtype=torch.float32))

    def forward(self, input, z):
        position = self.point_net(input, z)

        triplane_XY = self.triplane_generator_XY(z)
        triplane_XY = triplane_XY.view(triplane_XY.shape[0], 1, 32, triplane_XY.shape[-2], triplane_XY.shape[-1])
        triplane_XZ = self.triplane_generator_XZ(z)
        triplane_XZ = triplane_XZ.view(triplane_XZ.shape[0], 1, 32, triplane_XZ.shape[-2], triplane_XZ.shape[-1])
        triplane_YZ = self.triplane_generator_YZ(z)
        triplane_YZ = triplane_YZ.view(triplane_YZ.shape[0], 1, 32, triplane_YZ.shape[-2], triplane_YZ.shape[-1])
        triplane = torch.cat([triplane_XY, triplane_XZ, triplane_YZ], dim=1)


        point_feature = self.encoder(triplane, position)

        point_feature = point_feature.view(point_feature.shape[0], point_feature.shape[1], 3, 32).mean(dim=2)

        if self.opt.xyzres:
            point_feature = torch.cat([point_feature, position], dim=-1)

        shape_property = self.shape_transform(point_feature)
        color_property = self.color_transform(point_feature)

        return torch.cat([position, shape_property, color_property], dim=-1)

    def encoder(self, triplane, coordinates, mode='bilinear', padding_mode='zeros'):
        assert padding_mode == 'zeros'
        N, n_planes, C, H, W = triplane.shape
        _, M, _ = coordinates.shape
        plane_features = triplane.view(N * n_planes, C, H, W)

        coordinates = coordinates / self.bound

        def project_onto_triplane(triplane, coordinates):
            N, M, C = coordinates.shape
            n_planes, _, _ = triplane.shape
            coordinates = coordinates.unsqueeze(1).expand(-1, n_planes, -1, -1).reshape(N * n_planes, M, 3)
            inv_triplane = torch.linalg.inv(triplane).unsqueeze(0).expand(N, -1, -1, -1).reshape(N * n_planes, 3, 3)
            projections = torch.bmm(coordinates, inv_triplane)
            return projections[..., :2]

        projected_coordinates = project_onto_triplane(self.plane_axes, coordinates).unsqueeze(1)
        output_features = torch.nn.functional.grid_sample(plane_features, projected_coordinates.float(), mode=mode,
                                                          padding_mode=padding_mode, align_corners=False).view(N, n_planes, C, -1).view(N, n_planes*C, -1).transpose(1, 2)
        return output_features



