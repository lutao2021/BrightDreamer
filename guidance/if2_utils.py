from transformers import logging
from diffusers import IFPipeline, DDPMScheduler, DiffusionPipeline

# suppress partial model loading warning
logging.set_verbosity_error()

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.cuda.amp import custom_bwd, custom_fwd
from guidance.perpneg_utils import weighted_perpendicular_aggregator


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    #torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = True


class IF2(nn.Module):
    def __init__(self, device, vram_O, t_range=[0.02, 0.98]):
        super().__init__()

        self.device = device

        print(f'[INFO] loading DeepFloyd IF-II-XL...')

        model_key = "DeepFloyd/IF-II-L-v1.0"

        is_torch2 = torch.__version__[0] == '2'

        pipe = DiffusionPipeline.from_pretrained(
            model_key, text_encoder=None, variant="fp16", torch_dtype=torch.float16
        )

        pipe2 = IFPipeline.from_pretrained("DeepFloyd/IF-I-XL-v1.0", variant="fp16", torch_dtype=torch.float16)

        if vram_O:
            pipe.unet.to(memory_format=torch.channels_last)
            pipe.enable_attention_slicing(1)
            pipe.enable_model_cpu_offload()
        else:
            pipe.to(device)

        self.unet = pipe.unet
        self.tokenizer = pipe2.tokenizer
        self.text_encoder = pipe2.text_encoder.to(device)
        self.unet = pipe.unet
        self.scheduler = pipe.scheduler

        self.pipe = pipe

        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.min_step = int(self.num_train_timesteps * t_range[0])
        self.max_step = int(self.num_train_timesteps * t_range[1])
        self.alphas = self.scheduler.alphas_cumprod.to(self.device) # for convenience

        print(f'[INFO] loaded DeepFloyd IF-II-XL!')

    @torch.no_grad()
    def get_text_embeds(self, prompt):
        # prompt: [str]

        # TODO: should I add the preprocessing at https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/deepfloyd_if/pipeline_if.py#LL486C10-L486C28
        prompt = self.pipe._text_preprocessing(prompt, clean_caption=False)
        inputs = self.tokenizer(prompt, padding='max_length', max_length=77, truncation=True, add_special_tokens=True, return_tensors='pt')
        embeddings = self.text_encoder(inputs.input_ids.to(self.device))[0]

        return embeddings


    def train_step(self, text_embeddings, pred_rgb, guidance_scale=100, grad_scale=1):

        # [0, 1] to [-1, 1] and make sure shape is [64, 64]
        images = F.interpolate(pred_rgb, (256, 256), mode='bilinear', align_corners=False) * 2 - 1

        # timestep ~ U(0.02, 0.98) to avoid very high/low noise level
        t = torch.randint(self.min_step, self.max_step + 1, (images.shape[0],), dtype=torch.long, device=self.device)

        # predict the noise residual with unet, NO grad!
        with torch.no_grad():

            max_t = torch.full((pred_rgb.shape[0],), self.max_step, dtype=torch.long, device=self.device)

            # add noise
            noise = torch.randn_like(images)
            images_noisy = self.scheduler.add_noise(images, noise, t)

            intermediate_images = self.pipe.prepare_intermediate_images(
                pred_rgb.shape[0] * 1,
                self.unet.config.in_channels // 2,
                256,
                256,
                text_embeddings.dtype,
                self.device,
                None,
            )


            # pred noise
            model_input = torch.cat([intermediate_images, images_noisy], dim=1)
            model_input = torch.cat([model_input] * 2)
            model_input = self.scheduler.scale_model_input(model_input, t)
            tt = torch.cat([t] * 2)
            max_tt = torch.cat([max_t] * 2)
            noise_pred = self.unet(model_input, tt, encoder_hidden_states=text_embeddings, class_labels=max_tt).sample
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred_uncond, _ = noise_pred_uncond.split(model_input.shape[1] // 2, dim=1)
            noise_pred_text, predicted_variance = noise_pred_text.split(model_input.shape[1] // 2, dim=1)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # TODO: how to use the variance here?
            # noise_pred = torch.cat([noise_pred, predicted_variance], dim=1)

        # w(t), sigma_t^2
        w = (1 - self.alphas[t])
        grad = grad_scale * w[:, None, None, None] * (noise_pred - noise)
        grad = torch.nan_to_num(grad)

        targets = (images - grad).detach()
        loss = 0.5 * F.mse_loss(images.float(), targets, reduction='sum') / images.shape[0]

        return loss

    def train_step_perpneg(self, text_embeddings, weights, pred_rgb, guidance_scale=100, grad_scale=1):

        B = pred_rgb.shape[0]
        K = (text_embeddings.shape[0] // B) - 1 # maximum number of prompts        

        images = self.pipe.preprocess_image(pred_rgb, 1, pred_rgb.device)
        images = F.interpolate(images, (256, 256), mode='bilinear', align_corners=False) * 2 - 1

        # timestep ~ U(0.02, 0.98) to avoid very high/low noise level
        t = torch.randint(self.min_step, self.max_step + 1, (1,), dtype=torch.long, device=self.device).repeat(images.shape[0])

        # predict the noise residual with unet, NO grad!
        with torch.no_grad():

            max_t = torch.full((pred_rgb.shape[0],), self.max_step, dtype=torch.long, device=self.device)

            # add noise
            noise = torch.randn_like(images)
            images_noisy = self.scheduler.add_noise(images, noise, t)

            # pred noise
            model_input = torch.cat([images_noisy, images_noisy], dim=1)
            model_input = torch.cat([model_input] * (1 + K))
            model_input = self.scheduler.scale_model_input(model_input, t)
            tt = torch.cat([t] * (1 + K))
            max_tt = torch.cat([max_t] * (1 + K))
            unet_output = self.unet(model_input, tt, encoder_hidden_states=text_embeddings, class_labels=max_tt).sample
            noise_pred_uncond, noise_pred_text = unet_output[:B], unet_output[B:]
            noise_pred_uncond, _ = noise_pred_uncond.split(model_input.shape[1] // 2, dim=1)
            noise_pred_text, predicted_variance = noise_pred_text.split(model_input.shape[1] // 2, dim=1)
            # noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            delta_noise_preds = noise_pred_text - noise_pred_uncond.repeat(K, 1, 1, 1)
            noise_pred = noise_pred_uncond + guidance_scale * weighted_perpendicular_aggregator(delta_noise_preds, weights, B)



        # w(t), sigma_t^2
        w = (1 - self.alphas[t])
        grad = grad_scale * w[:, None, None, None] * (noise_pred - noise)
        grad = torch.nan_to_num(grad)

        targets = (images - grad).detach()
        loss = 0.5 * F.mse_loss(images.float(), targets, reduction='sum') / images.shape[0]

        return loss





if __name__ == '__main__':

    import argparse
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser()
    parser.add_argument('prompt', type=str)
    parser.add_argument('--negative', default='', type=str)
    parser.add_argument('--vram_O', action='store_true', help="optimization for low VRAM usage")
    parser.add_argument('-H', type=int, default=64)
    parser.add_argument('-W', type=int, default=64)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--steps', type=int, default=50)
    opt = parser.parse_args()

    seed_everything(opt.seed)

    device = torch.device('cuda')

    sd = IF2(device, opt.vram_O)

    imgs = sd.prompt_to_img(opt.prompt, opt.negative, opt.H, opt.W, opt.steps)

    # visualize image
    plt.imshow(imgs[0])
    plt.show()




