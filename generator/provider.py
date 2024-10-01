import random
import math
import numpy as np

from typing import NamedTuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.nn as nn


def safe_normalize(x, eps=1e-20):
    return x / torch.sqrt(torch.clamp(torch.sum(x * x, -1, keepdim=True), min=eps))


def gen_random_pos(size, param_range, gamma=1):
    lower, higher = param_range[0], param_range[1]

    mid = lower + (higher - lower) * 0.5
    radius = (higher - lower) * 0.5

    rand_ = torch.rand(size)  # 0, 1
    sign = torch.where(torch.rand(size) > 0.5, torch.ones(size) * -1., torch.ones(size))
    rand_ = sign * (rand_ ** gamma)

    return (rand_ * radius) + mid


def rand_poses(size, opt, radius_range=[1, 1.5], theta_range=[0, 120], phi_range=[0, 360], angle_overhead=30,
               angle_front=60, uniform_sphere_rate=0.5, rand_cam_gamma=1):
    ''' generate random poses from an orbit camera
    Args:
        size: batch size of generated poses.
        device: where to allocate the output.
        radius: camera radius
        theta_range: [min, max], should be in [0, pi]
        phi_range: [min, max], should be in [0, 2 * pi]
    Return:
        poses: [size, 4, 4]
    '''

    theta_range = np.array(theta_range) / 180 * np.pi
    phi_range = np.array(phi_range) / 180 * np.pi
    angle_overhead = angle_overhead / 180 * np.pi
    angle_front = angle_front / 180 * np.pi

    # radius = torch.rand(size) * (radius_range[1] - radius_range[0]) + radius_range[0]
    radius = gen_random_pos(size, radius_range)

    if random.random() < uniform_sphere_rate:
        unit_centers = F.normalize(
            torch.stack([
                torch.randn(size),
                torch.abs(torch.randn(size)),
                torch.randn(size),
            ], dim=-1), p=2, dim=1
        )
        thetas = torch.acos(unit_centers[:, 1])
        phis = torch.atan2(unit_centers[:, 0], unit_centers[:, 2])
        phis[phis < 0] += 2 * np.pi
        centers = unit_centers * radius.unsqueeze(-1)
    else:
        thetas = gen_random_pos(size, theta_range, rand_cam_gamma)
        phis = gen_random_pos(size, phi_range, rand_cam_gamma)
        phis[phis < 0] += 2 * np.pi

        centers = torch.stack([
            radius * torch.sin(thetas) * torch.sin(phis),
            radius * torch.sin(thetas) * torch.cos(phis),
            radius * torch.cos(thetas),
        ], dim=-1)  # [B, 3]

    targets = 0

    # jitters
    if opt.jitter_pose:
        jit_center = opt.jitter_center  # 0.015  # was 0.2
        jit_target = opt.jitter_target
        centers += torch.rand_like(centers) * jit_center - jit_center / 2.0
        targets += torch.randn_like(centers) * jit_target

    # lookat
    forward_vector = safe_normalize(centers - targets)
    up_vector = torch.FloatTensor([0, 0, 1]).unsqueeze(0).repeat(size, 1)
    # up_vector = torch.FloatTensor([0, 0, 1]).unsqueeze(0).repeat(size, 1)
    right_vector = safe_normalize(torch.cross(forward_vector, up_vector, dim=-1))

    if opt.jitter_pose:
        up_noise = torch.randn_like(up_vector) * opt.jitter_up
    else:
        up_noise = 0

    up_vector = safe_normalize(torch.cross(right_vector, forward_vector, dim=-1) + up_noise)  # forward_vector

    poses = torch.eye(4, dtype=torch.float).unsqueeze(0).repeat(size, 1, 1)
    poses[:, :3, :3] = torch.stack((-right_vector, up_vector, forward_vector), dim=-1)  # up_vector
    poses[:, :3, 3] = centers

    # back to degree
    thetas = thetas / np.pi * 180
    phis = phis / np.pi * 180

    return poses.numpy(), thetas.numpy(), phis.numpy(), radius.numpy()


def circle_poses(radius=torch.tensor([3.2]), theta=torch.tensor([60]), phi=torch.tensor([0]), angle_overhead=30,
                 angle_front=60):
    theta = theta / 180 * np.pi
    phi = phi / 180 * np.pi
    angle_overhead = angle_overhead / 180 * np.pi
    angle_front = angle_front / 180 * np.pi

    centers = torch.stack([
        radius * torch.sin(theta) * torch.sin(phi),
        radius * torch.sin(theta) * torch.cos(phi),
        radius * torch.cos(theta),
    ], dim=-1)  # [B, 3]

    # lookat
    forward_vector = safe_normalize(centers)
    up_vector = torch.FloatTensor([0, 0, 1]).unsqueeze(0).repeat(len(centers), 1)
    right_vector = safe_normalize(torch.cross(forward_vector, up_vector, dim=-1))
    up_vector = safe_normalize(torch.cross(right_vector, forward_vector, dim=-1))

    poses = torch.eye(4, dtype=torch.float).unsqueeze(0).repeat(len(centers), 1, 1)
    poses[:, :3, :3] = torch.stack((-right_vector, up_vector, forward_vector), dim=-1)
    poses[:, :3, 3] = centers

    return poses.numpy()


def focal2fov(focal, pixels):
    return 2 * math.atan(pixels / (2 * focal))


def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))


def GenerateRandomCameras(opt, size=2000, SSAA=True):
    # random pose on the fly
    poses, thetas, phis, radius = rand_poses(size, opt, radius_range=opt.radius_range, theta_range=opt.theta_range,
                                             phi_range=opt.phi_range,
                                             angle_overhead=opt.angle_overhead, angle_front=opt.angle_front,
                                             uniform_sphere_rate=opt.uniform_sphere_rate,
                                             rand_cam_gamma=1.0)
    # delta polar/azimuth/radius to default view
    delta_polar = thetas - opt.default_polar
    delta_azimuth = phis - opt.default_azimuth
    delta_azimuth[delta_azimuth > 180] -= 360  # range in [-180, 180]
    delta_radius = radius - opt.default_radius
    # random focal
    fov = random.random() * (opt.fovy_range[1] - opt.fovy_range[0]) + opt.fovy_range[0]

    cam_infos = []

    if SSAA:
        # ssaa = opt.SSAA
        ssaa = 1  # TODO
    else:
        ssaa = 1

    image_h = opt.image_h * ssaa
    image_w = opt.image_w * ssaa

    # generate specific data structure
    for idx in range(size):
        matrix = np.linalg.inv(poses[idx])
        R = -np.transpose(matrix[:3, :3])
        R[:, 0] = -R[:, 0]
        T = -matrix[:3, 3]
        fovy = focal2fov(fov2focal(fov, image_h), image_w)
        FovY = fovy
        FovX = fov

        cam_infos.append(RandCameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, width=image_w,
                                        height=image_h, delta_polar=delta_polar[idx],
                                        delta_azimuth=delta_azimuth[idx], delta_radius=delta_radius[idx]))
    return cam_infos


def GenerateCircleCameras(opt, size=8, render45=False):
    # random focal
    fov = opt.default_fovy
    cam_infos = []
    # generate specific data structure
    for idx in range(size):
        thetas = torch.FloatTensor([opt.default_polar])
        phis = torch.FloatTensor([(idx / size) * 360])
        radius = torch.FloatTensor([opt.default_radius])
        # random pose on the fly
        poses = circle_poses(radius=radius, theta=thetas, phi=phis, angle_overhead=opt.angle_overhead,
                             angle_front=opt.angle_front)
        matrix = np.linalg.inv(poses[0])
        R = -np.transpose(matrix[:3, :3])
        R[:, 0] = -R[:, 0]
        T = -matrix[:3, 3]
        fovy = focal2fov(fov2focal(fov, opt.image_h), opt.image_w)
        FovY = fovy
        FovX = fov

        # delta polar/azimuth/radius to default view
        delta_polar = thetas - opt.default_polar
        delta_azimuth = phis - opt.default_azimuth
        delta_azimuth[delta_azimuth > 180] -= 360  # range in [-180, 180]
        delta_radius = radius - opt.default_radius
        cam_infos.append(RandCameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, width=opt.image_w,
                                        height=opt.image_h, delta_polar=delta_polar, delta_azimuth=delta_azimuth,
                                        delta_radius=delta_radius))
    if render45:
        for idx in range(size):
            thetas = torch.FloatTensor([opt.default_polar * 2 // 3])
            phis = torch.FloatTensor([(idx / size) * 360])
            radius = torch.FloatTensor([opt.default_radius])
            # random pose on the fly
            poses = circle_poses(radius=radius, theta=thetas, phi=phis, angle_overhead=opt.angle_overhead,
                                 angle_front=opt.angle_front)
            matrix = np.linalg.inv(poses[0])
            R = -np.transpose(matrix[:3, :3])
            R[:, 0] = -R[:, 0]
            T = -matrix[:3, 3]
            fovy = focal2fov(fov2focal(fov, opt.image_h), opt.image_w)
            FovY = fovy
            FovX = fov

            # delta polar/azimuth/radius to default view
            delta_polar = thetas - opt.default_polar
            delta_azimuth = phis - opt.default_azimuth
            delta_azimuth[delta_azimuth > 180] -= 360  # range in [-180, 180]
            delta_radius = radius - opt.default_radius
            cam_infos.append(RandCameraInfo(uid=idx + size, R=R, T=T, FovY=FovY, FovX=FovX, width=opt.image_w,
                                            height=opt.image_h, delta_polar=delta_polar, delta_azimuth=delta_azimuth,
                                            delta_radius=delta_radius))
    return cam_infos


def GenerateCircleCameras2(opt, size=8, render45=False, radius=2.8):
    # random focal
    fov = opt.default_fovy
    cam_infos = []
    # generate specific data structure
    for idx in range(size):
        thetas = torch.FloatTensor([opt.default_polar])
        phis = torch.FloatTensor([(idx / size) * 360])
        radius = torch.FloatTensor([radius])
        # random pose on the fly
        poses = circle_poses(radius=radius, theta=thetas, phi=phis, angle_overhead=opt.angle_overhead,
                             angle_front=opt.angle_front)
        matrix = np.linalg.inv(poses[0])
        R = -np.transpose(matrix[:3, :3])
        R[:, 0] = -R[:, 0]
        T = -matrix[:3, 3]
        fovy = focal2fov(fov2focal(fov, opt.image_h), opt.image_w)
        FovY = fovy
        FovX = fov

        # delta polar/azimuth/radius to default view
        delta_polar = thetas - opt.default_polar
        delta_azimuth = phis - opt.default_azimuth
        delta_azimuth[delta_azimuth > 180] -= 360  # range in [-180, 180]
        delta_radius = radius - opt.default_radius
        cam_infos.append(RandCameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, width=opt.image_w,
                                        height=opt.image_h, delta_polar=delta_polar, delta_azimuth=delta_azimuth,
                                        delta_radius=delta_radius))
    if render45:
        for idx in range(size):
            thetas = torch.FloatTensor([opt.default_polar * 2 // 3])
            phis = torch.FloatTensor([(idx / size) * 360])
            radius = torch.FloatTensor([radius])
            # random pose on the fly
            poses = circle_poses(radius=radius, theta=thetas, phi=phis, angle_overhead=opt.angle_overhead,
                                 angle_front=opt.angle_front)
            matrix = np.linalg.inv(poses[0])
            R = -np.transpose(matrix[:3, :3])
            R[:, 0] = -R[:, 0]
            T = -matrix[:3, 3]
            fovy = focal2fov(fov2focal(fov, opt.image_h), opt.image_w)
            FovY = fovy
            FovX = fov

            # delta polar/azimuth/radius to default view
            delta_polar = thetas - opt.default_polar
            delta_azimuth = phis - opt.default_azimuth
            delta_azimuth[delta_azimuth > 180] -= 360  # range in [-180, 180]
            delta_radius = radius - opt.default_radius
            cam_infos.append(RandCameraInfo(uid=idx + size, R=R, T=T, FovY=FovY, FovX=FovX, width=opt.image_w,
                                            height=opt.image_h, delta_polar=delta_polar, delta_azimuth=delta_azimuth,
                                            delta_radius=delta_radius))
    return cam_infos


def GenerateCubePlaneCameras(opt):
    # random focal
    fov = opt.default_fovy
    cam_infos = []
    theta_phis = [[0, 0], [90, 0], [90, 90], [90, 180], [90, 270], [180, 0]]
    radius = 2.8

    for idx in range(1, 6):
        thetas = torch.FloatTensor([theta_phis[idx][0]])
        phis = torch.FloatTensor([theta_phis[idx][1]])
        radius = torch.FloatTensor([radius])
        x, y, z = radius * torch.sin(thetas) * torch.sin(phis), radius * torch.sin(thetas) * torch.cos(
            phis), radius * torch.cos(thetas)
        ### calculate R and T
        forward = np.concatenate([x, y, z])
        forward = -forward / np.linalg.norm(forward)  # 归一化并取负值

        up = np.array([0, 1, 0])
        right = np.cross(up, forward)
        right = right / np.linalg.norm(right)
        up = np.cross(forward, right)
        R = np.column_stack((right, up, forward))
        T = np.concatenate([x, y, z])

        fovy = focal2fov(fov2focal(fov, opt.image_h), opt.image_w)
        FovY = fovy
        FovX = fov

        # delta polar/azimuth/radius to default view
        delta_polar = thetas - opt.default_polar
        delta_azimuth = phis - opt.default_azimuth
        delta_azimuth[delta_azimuth > 180] -= 360  # range in [-180, 180]
        delta_radius = radius - opt.default_radius
        cam_infos.append(RandCameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, width=opt.image_w,
                                        height=opt.image_h, delta_polar=delta_polar, delta_azimuth=delta_azimuth,
                                        delta_radius=delta_radius))
    return cam_infos


def cameraList_from_RcamInfos(cam_infos, resolution_scale, opt, SSAA=False):
    camera_list = []

    for id, c in enumerate(cam_infos):
        camera_list.append(loadRandomCam(opt, id, c, resolution_scale, SSAA=SSAA))

    return camera_list


def loadRandomCam(opt, id, cam_info, resolution_scale, SSAA=False):
    return RCamera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T,
                   FoVx=cam_info.FovX, FoVy=cam_info.FovY, delta_polar=cam_info.delta_polar,
                   delta_azimuth=cam_info.delta_azimuth, delta_radius=cam_info.delta_radius, opt=opt,
                   uid=id, data_device=opt.device, SSAA=SSAA)


def getWorld2View2(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)


def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P


class RandCameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    width: int
    height: int
    delta_polar: np.array
    delta_azimuth: np.array
    delta_radius: np.array


class RCamera(nn.Module):
    def __init__(self, colmap_id, R, T, FoVx, FoVy, uid, delta_polar, delta_azimuth, delta_radius, opt,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device="cuda", SSAA=False
                 ):
        super(RCamera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.delta_polar = delta_polar
        self.delta_azimuth = delta_azimuth
        self.delta_radius = delta_radius
        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device")
            self.data_device = torch.device("cuda")

        self.zfar = 100.0
        self.znear = 0.01

        if SSAA:
            # ssaa = opt.SSAA
            ssaa = 1  # TODO
        else:
            ssaa = 1

        self.image_width = opt.image_w * ssaa
        self.image_height = opt.image_h * ssaa

        self.trans = trans
        self.scale = scale

        RT = torch.tensor(getWorld2View2(R, T, trans, scale))
        self.world_view_transform = RT.transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx,
                                                     fovY=self.FoVy).transpose(0, 1).cuda()
        self.full_proj_transform = (
            self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]


class GasussianDataset:
    def __init__(self, opt, device, guidance, type='train'):
        # TODO: directly use the text encoder rather than the guidance
        super().__init__()

        self.opt = opt
        self.device = device
        self.type = type  # train, val, test

        if self.type == 'val':
            self.num_view = 8
        else:
            self.num_view = 100

        self.training = self.type in ['train', 'all']

        if opt.cache_path is None:
            self.use_cache = False
            self.guidance = guidance
        else:
            self.use_cache = True

        if opt.prompts_set == 'vehicle':
            self.texts = []
            with open('./vehicle.txt', 'r') as f:
                texts = f.readlines()
            for text in texts:
                if len(text.strip().split(' ')) < 75:
                    self.texts.append(text.strip().strip('.'))

            print(len(self.texts))
            if self.type == 'train':
                if opt.cache_path is not None:
                    self.texts_embeddings = torch.load(self.opt.cache_path)
                    assert len(self.texts) == len(self.texts_embeddings)
            else:
                self.texts = ['Electric luxury SUV, deep purple, spacious, advanced tech',
                              'Electric luxury SUV, light yellow, spacious, advanced tech',
                              'Vintage pickup, sky blue, rugged appeal, classic functionality',
                              'Electric luxury SUV, light yellow, spacious, advanced tech',
                              'Luxury roadster, sky blue, unmatched elegance, superior driving experience']

            if self.type != 'train':
                self.texts_embeddings = []
                for text in self.texts:
                    embeddings = {}
                    embeddings['default'] = guidance.get_text_embeds([text]).cpu()
                    for d in ['front', 'side', 'back']:
                        embeddings[d] = guidance.get_text_embeds([f"{text}, {d} view"]).cpu()
                    self.texts_embeddings.append(embeddings)

        elif opt.prompts_set == 'daily_life':
            self.texts = []
            with open('./daily_life.txt', 'r') as f:
                texts = f.readlines()
            for text in texts:
                if len(text.strip().split(' ')) < 75:
                    self.texts.append(text.strip())

            print(len(self.texts))
            if self.type == 'train':
                self.texts = self.texts
                if opt.cache_path is not None:
                    self.texts_embeddings = torch.load(self.opt.cache_path)
                    assert len(self.texts) == len(self.texts_embeddings)
            else:
                self.texts = ['a man wearing a hat is mowing the lawn',
                              'A handsome man wearing a leather jacket is riding a motorcycle',
                              'A stylish woman in a long dress is climbing a mountain',
                              'A glamorous woman in a cocktail dress is dancing at a fancy party',
                              'a woman wearing a backpack is climbing a mountain']
            if self.type != 'train':
                self.texts_embeddings = []
                for text in self.texts:
                    embeddings = {}
                    embeddings['default'] = guidance.get_text_embeds([text]).cpu()
                    for d in ['front', 'side', 'back']:
                        embeddings[d] = guidance.get_text_embeds([f"{text}, {d} view"]).cpu()
                    self.texts_embeddings.append(embeddings)


        elif opt.prompts_set == 'animal':
            self.texts = []
            species = ['wolf', 'dog', 'panda', 'fox', 'civet', 'cat', 'red panda', 'teddy bear', 'rabbit', 'koala']
            item = ['in a bathtub', 'on a stone', 'on books', 'on a table', 'on the lawn', 'in a basket', 'null']
            gadget = ['a tie', 'a cape', 'sunglasses', 'a scarf', 'null']
            hat = ['beret', 'beanie', 'cowboy hat', 'straw hat', 'baseball cap', 'tophat', 'party hat', 'sombrero',
                   'null']
            for s in species:
                for i in item:
                    for g in gadget:
                        for h in hat:
                            if i == 'null':
                                self.texts.append(f'a {s} wearing {g} and wearing a {h}')
                            elif g == 'null':
                                self.texts.append(f'a {s} sitting {i} and wearing a {h}')
                            elif h == 'null':
                                self.texts.append(f'a {s} sitting {i} and wearing {g}')
                            else:
                                self.texts.append(f'a {s} sitting {i} and wearing {g} and wearing a {h}')

            print(len(self.texts))
            if self.type == 'train':
                self.texts = self.texts
                if opt.cache_path is not None:
                    self.texts_embeddings = torch.load(self.opt.cache_path)
                    assert len(self.texts) == len(self.texts_embeddings)
            if self.type != 'train':
                self.texts = ['a dog wearing a tie and wearing a party hat',
                              'a koala sitting on a stone and wearing a cape and wearing a cowboy hat',
                              'a panda sitting in a basket and wearing a straw hat',
                              'a panda sitting in a bathtub and wearing a beanie',
                              'a rabbit sitting on a stone and wearing a tie',
                              'a teddy bear sitting on a stone and wearing a scarf and wearing a flat cap',
                              'a teddy bear sitting on books and wearing a scarf and wearing a flat cap',
                              'a wolf wearing a tie and wearing a party hat']
                self.texts_embeddings = []
                for text in self.texts:
                    embeddings = {}
                    embeddings['default'] = guidance.get_text_embeds([text]).cpu()
                    for d in ['front', 'side', 'back']:
                        embeddings[d] = guidance.get_text_embeds([f"{text}, {d} view"]).cpu()
                    self.texts_embeddings.append(embeddings)
                    
        elif opt.prompts_set == 'mix':
            self.texts = []
            with open('./mix.txt', 'r') as f:
                texts = f.readlines()
            for text in texts:
                if len(text.strip().split(' ')) < 75:
                    self.texts.append(text.strip())

            print(len(self.texts))
            if self.type == 'train':
                self.texts = self.texts
                if opt.cache_path is not None:
                    self.texts_embeddings = torch.load(self.opt.cache_path)
                    assert len(self.texts) == len(self.texts_embeddings)
            else:
                self.texts = ['Electric luxury SUV, deep purple, spacious, advanced tech',
                              'Electric luxury SUV, light yellow, spacious, advanced tech',
                              'Vintage pickup, sky blue, rugged appeal, classic functionality',
                              'a man wearing a hat is mowing the lawn',
                              'A handsome man wearing a leather jacket is riding a motorcycle',
                              'A stylish woman in a long dress is climbing a mountain',
                              'A glamorous woman in a cocktail dress is dancing at a fancy party',
                              'a woman wearing a backpack is climbing a mountain',
                              'a teddy bear sitting on books and wearing a scarf and wearing a flat cap',
                              'a wolf wearing a tie and wearing a party hat']
            if self.type != 'train':
                self.texts_embeddings = []
                for text in self.texts:
                    embeddings = {}
                    embeddings['default'] = guidance.get_text_embeds([text]).cpu()
                    for d in ['front', 'side', 'back']:
                        embeddings[d] = guidance.get_text_embeds([f"{text}, {d} view"]).cpu()
                    self.texts_embeddings.append(embeddings)
        

        self.size = len(self.texts)
        self.uncond_embed = guidance.get_text_embeds(['']).cpu()

    def __len__(self):
        return self.size

    def get_text_embeddings_gpu(self, index):
        if self.use_cache:
            texts_embedding = {}
            texts_embedding['default'] = self.texts_embeddings[index]['default'].to(self.device)
            texts_embedding['uncond'] = self.uncond_embed.to(self.device)
            for d in ['front', 'side', 'back']:
                texts_embedding[d] = self.texts_embeddings[index][d].to(self.device)
        else:
            texts_embedding = {}
            texts_embedding['default'] = self.guidance.get_text_embeds([self.texts[index]]).to(self.device)
            texts_embedding['uncond'] = self.uncond_embed.to(self.device)
            for d in ['front', 'side', 'back']:
                texts_embedding[d] = self.guidance.get_text_embeds([f"{self.texts[index]}, {d} view"]).to(self.device)

        return texts_embedding

    def collate(self, index):
        # collate can help control text batch size and camera batch size easier

        # TODO: clarify the amout of views or texts?

        B = len(index)  # text batch size
        C = self.opt.c_batch_size  # camera batch size
        data = []
        for i in range(B):
            if self.training:
                rand_train_cameras = GenerateRandomCameras(self.opt, C, SSAA=True)
                cameras = cameraList_from_RcamInfos(rand_train_cameras, 1.0,
                                                    self.opt, SSAA=True)
                data.append({'cameras': cameras, 'text_embeddings': self.get_text_embeddings_gpu(index[i]),
                             'text': self.texts[index[i]]})
            else:
                eval_cameras = GenerateCircleCameras(self.opt, self.num_view, render45=False)
                cameras = cameraList_from_RcamInfos(eval_cameras, 1.0,
                                                    self.opt, SSAA=True)
                data.append({'cameras': cameras, 'text_embeddings': self.get_text_embeddings_gpu(index[i]),
                             'text': self.texts[index[i]]})

        return data

    def dataloader(self):
        if self.training:
            batch_size = self.opt.batch_size
        else:
            batch_size = 1

        if (self.opt.num_gpus > 1 and self.training) or (self.type == 'test' and self.opt.num_gpus > 1):
            ddp_sampler = torch.utils.data.distributed.DistributedSampler(list(range(self.size)), shuffle=self.training)
            loader = DataLoader(list(range(self.size)), batch_size=batch_size, collate_fn=self.collate, shuffle=False,
                                num_workers=0, sampler=ddp_sampler)
        else:
            loader = DataLoader(list(range(self.size)), batch_size=batch_size, collate_fn=self.collate,
                                shuffle=self.training, num_workers=0)

        loader._data = self
        return loader
