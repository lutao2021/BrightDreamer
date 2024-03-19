import torch
import torch.nn as nn
from .generator import Generator
from generator.gaussian_utils.gaussian_model import GaussianModel
from generator.gaussian_utils.gaussian_renderer import render

C0 = 0.28209479177387814
def RGB2SH(rgb):
    return (rgb - 0.5) / C0

class PipelineParams:
    def __init__(self):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False

class BrightDreamer(nn.Module):
    def __init__(self, opt):
        super().__init__()

        self.opt=opt
        self.sh_degree = 0
        self.img_channel = 3 + 1 + 4 + 3 + 3 * (self.sh_degree + 1) ** 2
        self.grid_resolution = 64
        self.bound = 1.0
        self.free_distance = self.opt.free_distance
        self.generator = Generator(opt=opt)
        self.pp = PipelineParams()
        self.register_buffer('background', torch.ones(3))

        self.register_buffer('anchor_position', torch.stack(torch.meshgrid(torch.arange(-self.bound, self.bound, self.bound*2/self.grid_resolution),
                                                                         torch.arange(-self.bound, self.bound, self.bound*2/self.grid_resolution),
                                                                         torch.arange(-self.bound, self.bound, self.bound*2/self.grid_resolution)), dim=3).reshape(-1,3).contiguous())


    def gaussian_generate(self, text_embeddings):

        B = text_embeddings.shape[0]
        input = self.anchor_position.unsqueeze(0).repeat(B, 1, 1)  # (B, 128*128*128, 3)
        gaussians_property = self.generator(input, text_embeddings)  # (B, 128*128*128, 14)

        gaussians_property = gaussians_property.to(torch.float32)
        gaussian_list = []  # (B,)
        for i in range(B):

            gaussian = GaussianModel(self.sh_degree)
            gaussian._xyz = gaussians_property[i, :, 0:3]
            gaussian._opacity = gaussians_property[i, :, 3:4]
            gaussian._rotation = gaussians_property[i, :, 4:8]
            gaussian._scaling = (6*torch.sigmoid(gaussians_property[i, :, 8:11]) - 9)
            gaussian._features_dc = (3.545*torch.sigmoid(gaussians_property[i, :, 11:14].reshape(-1, 1, 3)) - 1.7725)
            gaussian._features_rest = gaussians_property[i, :, 14:].reshape(-1, 15, 3)
            gaussian_list.append(gaussian)

        return gaussian_list

    def render(self, gaussians, views):
        B = len(gaussians)
        C = len(views[0])
        rgbs = []
        for i in range(B):
            gaussian = gaussians[i]
            for j in range(C):
                view = views[i][j]
                render_pkg = render(view, gaussian, self.pp, self.background)
                rgb = render_pkg['render']
                rgbs.append(rgb)
        rgbs = torch.stack(rgbs, dim=0)
        return {'rgbs': rgbs}

    def forward(self, text_zs, cameras):
        gaussians = self.gaussian_generate(text_zs)
        outputs = self.render(gaussians, cameras)
        return outputs

    def get_params(self, lr):

        params = [
            {'params': self.parameters(), 'lr': lr},
        ]
        return params