import torch
import torch.nn as nn
import argparse
import sys
import torch.distributed as dist
import os

from generator.provider import GasussianDataset
from trainer import *
from generator.BrightDreamer import BrightDreamer



def train(rank, opt):
    # dist init
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = opt.port
    opt.local_rank = rank
    if opt.num_gpus > 1:
        dist_backend = 'nccl'
        dist.init_process_group(backend=dist_backend, rank=rank, world_size=opt.num_gpus)
        seed = int(opt.seed)
        seed_everything(seed)
        print('args.local_rank: ', opt.local_rank)
        torch.cuda.set_device(opt.local_rank)
    else:
        if opt.seed is not None:
            seed_everything(int(opt.seed))

    opt.image_h = opt.h
    opt.image_w = opt.w

    print(opt)

    # dist device
    if opt.local_rank != -1:
        torch.cuda.set_device(opt.local_rank)
        device = torch.device(opt.local_rank)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    opt.device = device

    model = BrightDreamer(opt).to(device)

    # dist model
    if opt.num_gpus > 1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[opt.local_rank], output_device=opt.local_rank)
        seed_everything(opt.seed + opt.local_rank)
    else:
        seed_everything(opt.seed)



    if opt.num_gpus > 1:
        if opt.optim == 'adan':
            from optimizer import Adan

            # Adan usually requires a larger LR
            optimizer = lambda model: Adan(model.module.get_params(5 * opt.lr), eps=1e-8, weight_decay=2e-5,
                                           max_grad_norm=5.0, foreach=False)
        else:  # adam
            optimizer = lambda model: torch.optim.Adam(model.module.get_params(opt.lr), betas=(0.9, 0.99), eps=1e-15)

    else:
        if opt.optim == 'adan':
            from optimizer import Adan

            # Adan usually requires a larger LR
            optimizer = lambda model: Adan(model.get_params(5 * opt.lr), eps=1e-8, weight_decay=2e-5, max_grad_norm=5.0,
                                           foreach=False)
        else:  # adam
            optimizer = lambda model: torch.optim.Adam(model.get_params(opt.lr), betas=(0.9, 0.99), eps=1e-15)

    scheduler = lambda optimizer: optim.lr_scheduler.LambdaLR(optimizer, lambda iter: 1)  # fixed

    guidance = nn.ModuleDict()

    if 'SD' in opt.guidance:
        from guidance.sd_utils import StableDiffusion

        guidance['SD'] = StableDiffusion(device, opt.fp16, opt.vram_O, opt.sd_version, opt.hf_key, opt.t_range)

    if 'IF' in opt.guidance:
        from guidance.if_utils import IF

        guidance['IF'] = IF(device, opt.vram_O, opt.t_range)

    if 'IF2' in opt.guidance:
        from guidance.if2_utils import IF2

        guidance['IF2'] = IF2(device, opt.vram_O, opt.t_range)

    if opt.num_gpus > 1:
        trainer = Trainer(' '.join(sys.argv), 'BrightDreamer', opt, model, guidance, device=device, workspace=opt.workspace,
                          optimizer=optimizer, ema_decay=opt.ema_decay, fp16=opt.fp16, lr_scheduler=scheduler,
                          use_checkpoint=opt.ckpt, scheduler_update_every_step=True, local_rank=opt.local_rank)
    else:
        trainer = Trainer(' '.join(sys.argv), 'BrightDreamer', opt, model, guidance, device=device, workspace=opt.workspace,
                          optimizer=optimizer, ema_decay=opt.ema_decay, fp16=opt.fp16, lr_scheduler=scheduler,
                          use_checkpoint=opt.ckpt, scheduler_update_every_step=True)

    train_loader = GasussianDataset(opt, device=device, guidance=guidance[opt.guidance[0]], type='train').dataloader()

    valid_loader = GasussianDataset(opt, device=device, guidance=guidance[opt.guidance[0]], type='val').dataloader()

    test_loader = GasussianDataset(opt, device=device, guidance=guidance[opt.guidance[0]], type='test').dataloader()

    max_epoch = np.ceil(opt.iters / len(train_loader)).astype(np.int32)

    if not opt.test:
        trainer.train(train_loader, valid_loader, test_loader, max_epoch)
    else:
        trainer.test(test_loader)

    if opt.save_mesh:
        trainer.save_mesh()

if __name__ == '__main__':
    # See https://stackoverflow.com/questions/27433316/how-to-get-argparse-to-read-arguments-from-a-file-with-an-option-rather-than-pre
    class LoadFromFile (argparse.Action):
        def __call__ (self, parser, namespace, values, option_string = None):
            with values as f:
                # parse arguments in the file and store them in the target namespace
                parser.parse_args(f.read().split(), namespace)

    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument('--test', action='store_true', help="test mode")
    parser.add_argument('--eval_interval', type=int, default=1, help="evaluate on the valid set every interval epochs")
    parser.add_argument('--test_interval', type=int, default=500, help="test on the test set every interval epochs")
    parser.add_argument('--workspace', type=str, default='workspace')
    parser.add_argument('--seed', type=int, default=0)

    parser.add_argument('--guidance', type=str, nargs='*', default=['SD'], help='guidance model')
    parser.add_argument('--guidance_scale', type=float, default=100, help="diffusion model classifier-free guidance scale")

    ## Perp-Neg options
    parser.add_argument('--perpneg', action='store_true', help="use perp_neg")
    parser.add_argument('--negative_w', type=float, default=-2, help="The scale of the weights of negative prompts. A larger absolute value will help to avoid the Janus problem, but may cause flat faces. Vary between 0 to -4, depending on the prompt")
    parser.add_argument('--front_decay_factor', type=float, default=2, help="decay factor for the front prompt")
    parser.add_argument('--side_decay_factor', type=float, default=10, help="decay factor for the side prompt")

    ### training options
    parser.add_argument('--iters', type=int, default=10000, help="training iters")
    parser.add_argument('--lr', type=float, default=1e-3, help="max learning rate")
    parser.add_argument('--ckpt', type=str, default='latest', help="possible options are ['latest', 'scratch', 'best', 'latest_model']")
    parser.add_argument('--grad_clip', type=float, default=-1, help="clip grad of all grad to this limit, negative value disables it")
    parser.add_argument('--lambda_guidance', type=float, default=1, help="loss scale for SDS")

    # network generator
    parser.add_argument('--optim', type=str, default='adam', choices=['adan', 'adam'], help="optimizer")
    parser.add_argument('--sd_version', type=str, default='2.1', choices=['1.5', '2.0', '2.1'], help="stable diffusion version")
    parser.add_argument('--hf_key', type=str, default=None, help="hugging face Stable diffusion model key")
    # try this if CUDA OOM
    parser.add_argument('--fp16', action='store_true', help="use float16 for training")
    parser.add_argument('--vram_O', action='store_true', help="optimization for low VRAM usage")
    # rendering resolution in training, increase these for better quality / decrease these if CUDA OOM even if --vram_O enabled.
    parser.add_argument('--w', type=int, default=512, help="render width in training")
    parser.add_argument('--h', type=int, default=512, help="render height in training")
    parser.add_argument('--batch_size', type=int, default=8, help="batch_size of prompts")
    parser.add_argument('--c_batch_size', type=int, default=4, help="camera batch size for each prompt")

    ### dataset options
    parser.add_argument('--prompts_set', type=str, default='vehicle', choices=['vehicle', 'daily_life', 'animal', 'mix'], help="optimizer")
    parser.add_argument('--cache_path', type=str, default=None, help="optimizer")

    parser.add_argument('--bound', type=float, default=1, help="assume the scene is bounded in box(-bound, bound)")

    parser.add_argument('--radius_range', type=float, nargs='*', default=[3.5, 3.5], help="training camera radius range")
    parser.add_argument('--theta_range', type=float, nargs='*', default=[45, 105], help="training camera range along the polar angles (i.e. up and down). See advanced.md for details.")
    parser.add_argument('--phi_range', type=float, nargs='*', default=[-180, 180], help="training camera range along the azimuth angles (i.e. left and right). See advanced.md for details.")
    parser.add_argument('--fovy_range', type=float, nargs='*', default=[20, 20], help="training camera fovy range (tan value)")
    parser.add_argument('--jitter_pose', action='store_true', help="add jitters to the randomly sampled camera poses")
    parser.add_argument('--jitter_center', type=float, default=0.2, help="amount of jitter to add to sampled camera pose's center (camera location)")
    parser.add_argument('--jitter_target', type=float, default=0.2, help="amount of jitter to add to sampled camera pose's target (i.e. 'look-at')")
    parser.add_argument('--jitter_up', type=float, default=0.02, help="amount of jitter to add to sampled camera pose's up-axis (i.e. 'camera roll')")
    parser.add_argument('--uniform_sphere_rate', type=float, default=0, help="likelihood of sampling camera location uniformly on the sphere surface area")

    parser.add_argument('--default_radius', type=float, default=3.5, help="radius for the default view")
    parser.add_argument('--default_polar', type=float, default=90, help="polar for the default view")
    parser.add_argument('--default_azimuth', type=float, default=0, help="azimuth for the default view")
    parser.add_argument('--default_fovy', type=float, default=20, help="fovy for the default view (tan value)")

    parser.add_argument('--angle_overhead', type=float, default=30, help="[0, angle_overhead] is the overhead region")
    parser.add_argument('--angle_front', type=float, default=60, help="[0, angle_front] is the front region, [180, 180+angle_front] the back region, otherwise the side region.")
    parser.add_argument('--t_range', type=float, nargs='*', default=[0.02, 0.98], help="stable diffusion time steps range")

    ### debugging options
    parser.add_argument('--save_guidance', action='store_true', help="save images of the per-iteration NeRF renders, added noise, denoised (i.e. guidance), fully-denoised. Useful for debugging, but VERY SLOW and takes lots of memory!")
    parser.add_argument('--save_guidance_interval', type=int, default=10, help="save guidance every X step")

    parser.add_argument('--save_interval', type=int, default=1, help="ckpt save interval")
    parser.add_argument('--port', type=str, default='12355')
    parser.add_argument('--xyzres', action='store_true', help="xyz input for gaussian decoding")
    parser.add_argument('--free_distance', type=float, default=0.2, help="max deviation from anchor positions")
    parser.add_argument('--ema_decay', type=float, default=None)

    opt = parser.parse_args()

    if os.environ['CUDA_VISIBLE_DEVICES'].find(',') != -1:
        opt.num_gpus = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
        print('gpu:', os.environ['CUDA_VISIBLE_DEVICES'])
    else:
        opt.num_gpus = 1
    if opt.num_gpus > 1:
        torch.multiprocessing.spawn(train, nprocs=opt.num_gpus, args=(opt,), join=True)
    else:
        train(rank=0, opt=opt)


