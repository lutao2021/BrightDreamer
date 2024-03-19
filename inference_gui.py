from flask import Flask, request, render_template, send_from_directory
import os

import torch
import argparse
import imageio
from torch_ema import ExponentialMovingAverage

from generator.BrightDreamer import BrightDreamer
from generator.provider import GenerateCircleCameras, cameraList_from_RcamInfos
from diffusers import IFPipeline, StableDiffusionPipeline
from torchvision.utils import save_image



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str)

    # network params
    parser.add_argument('--bound', type=float, default=1, help="assume the scene is bounded in box(-bound, bound)")
    parser.add_argument('--xyzres', action='store_true', help="xyzres")
    parser.add_argument('--free_distance', type=float, default=0.2)

    # render params
    parser.add_argument('--radius_range', type=float, nargs='*', default=[3.5, 3.5], help="training camera radius range")
    parser.add_argument('--theta_range', type=float, nargs='*', default=[45, 105], help="training camera range along the polar angles (i.e. up and down). See advanced.md for details.")
    parser.add_argument('--phi_range', type=float, nargs='*', default=[-180, 180], help="training camera range along the azimuth angles (i.e. left and right). See advanced.md for details.")
    parser.add_argument('--fovy_range', type=float, nargs='*', default=[20, 20], help="training camera fovy range")
    parser.add_argument('--default_radius', type=float, default=3.5, help="radius for the default view")
    parser.add_argument('--default_polar', type=float, default=90, help="polar for the default view")
    parser.add_argument('--default_azimuth', type=float, default=0, help="azimuth for the default view")
    parser.add_argument('--default_fovy', type=float, default=20, help="fovy for the default view")
    parser.add_argument('--angle_overhead', type=float, default=30, help="[0, angle_overhead] is the overhead region")
    parser.add_argument('--angle_front', type=float, default=60,
                        help="[0, angle_front] is the front region, [180, 180+angle_front] the back region, otherwise the side region.")



    parser.add_argument('--w', type=int, default=512, help="render width for NeRF in training")
    parser.add_argument('--h', type=int, default=512, help="render height for NeRF in training")
    parser.add_argument('--ema', action='store_true', help="load ema weights")

    opt = parser.parse_args()
    return opt


opt = parse_args()
opt.xyzres = True
opt.fp16 = True
opt.image_h = opt.h
opt.image_w = opt.w
device = torch.device('cuda')
opt.device = device

generator = BrightDreamer(opt).to(device)
model_ckpt = torch.load(opt.model_path, map_location='cpu')
generator.load_state_dict(model_ckpt['model'])
if 'ema' in model_ckpt and opt.ema:
    ema = ExponentialMovingAverage(generator.parameters(), decay=0.99)
    ema.load_state_dict(model_ckpt['ema'])
    ema.copy_to()

pipe = IFPipeline.from_pretrained(
    "DeepFloyd/IF-I-XL-v1.0", variant="fp16",
    torch_dtype=torch.float16)
tokenizer = pipe.tokenizer
text_encoder = pipe.text_encoder.to(device)

# pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1", variant="fp16", torch_dtype=torch.float16)
# tokenizer = pipe.tokenizer
# text_encoder = pipe.text_encoder.to(device)

generator.eval()

app = Flask(__name__)

VIDEO_FOLDER = './video_cache'
app.config['VIDEO_FOLDER'] = VIDEO_FOLDER

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate_video():
    opt.prompt = request.form['text']
    camera_radius = request.form.get('cameraRadius', '3.5 ')
    opt.default_radius = float(camera_radius)
    default_polar = request.form.get('polarAngle', '90')
    opt.default_polar = float(default_polar)
    cameras = generate_camera_path(120, opt)[0:120]
    video_filename = generate_video_from_text(opt.prompt, cameras)
    return video_filename

@app.route('/videos/<filename>')
def video(filename):
    return send_from_directory(app.config['VIDEO_FOLDER'], filename)

# camera path generation
def generate_camera_path(num_view, opt):
    eval_cameras = GenerateCircleCameras(opt, num_view, render45=False)
    cameras = cameraList_from_RcamInfos(eval_cameras, 1.0, opt, SSAA=True)
    return cameras

def generate_video_from_text(prompt, cameras):
    inputs = tokenizer(prompt, padding='max_length', max_length=77, truncation=True, add_special_tokens=True, return_tensors='pt')
    embeddings = text_encoder(inputs.input_ids.to(device))[0]
    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=opt.fp16):
            gaussian_models = generator.gaussian_generate(embeddings)
        images = generator.render(gaussian_models, [cameras])['rgbs']
        images = images * 255
        images = images.permute(0, 2, 3, 1)

        save_path = './video_cache'
        os.makedirs(save_path, exist_ok=True)
        imageio.mimwrite(os.path.join(save_path, 'video.mp4'), images.to(torch.uint8).cpu(), fps=25, quality=8, macro_block_size=1)
    return 'video.mp4'

@app.route('/video_cache/<filename>')
def video_cache(filename):
    return send_from_directory(app.config['VIDEO_FOLDER'], filename)

@app.route('/3DGS_model_cache/<filename>')
def GS_model_cache(filename):
    print(opt.prompt)
    inputs = tokenizer(opt.prompt, padding='max_length', max_length=77, truncation=True, add_special_tokens=True, return_tensors='pt')
    embeddings = text_encoder(inputs.input_ids.to(device))[0]
    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=opt.fp16):
            gaussian_models = generator.gaussian_generate(embeddings)
    save_path = './video_cache'
    gaussian_models[0].save_ply(save_path + '/3DGS_model.ply')
    return send_from_directory(app.config['VIDEO_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=False)