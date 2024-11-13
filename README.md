# BrightDreamer: Generic 3D Gaussian Generative Framework for Fast Text-to-3D Synthesis

### [Project Page](vlislab22.github.io/BrightDreamer/) | [Arxiv](https://arxiv.org/abs/2403.11273) | [Weights](https://drive.google.com/drive/folders/14GOrlRbpROZw3SOLKqLCMIoYUTbL-lOb)

The official implementation of BrightDreamer: Generic 3D Gaussian Generative Framework for Fast Text-to-3D Synthesis.

If you find this work interesting or useful, please give me a ⭐!

User Interactive Demo

https://github.com/lutao2021/BrightDreamer/assets/114853034/f59a5143-d1d2-4a5e-bdae-c873c183a1d5

Interpolation Demonstration Demo

https://github.com/lutao2021/BrightDreamer/assets/114853034/336d67c4-df2b-4643-98d4-413a64942718


# Quick Start

* [Inference based on command](#2-infer-by-users-input-from-command-without-the-requirements-to-load-model-at-each-time-you-can-input-the-prompt-in-command-input-exit-to-quit)

* [Inference based on Web GUI](#3-gui-interactive-method-as-previous-demo-shows)

* [Training](#training)

If you have any questions about this project, please feel free to open an issue.

---

# Install
```bash
git clone https://github.com/lutao2021/BrightDreamer.git
cd BrightDreamer
conda create -n BrightDreamer python=3.9
conda activate BrightDreamer
```

You need to first install the suitable torch and torchvision according your environment. The version used in our experiments is 

```
torch==1.13.1+cu117   torchvision==0.14.1+cu117
```

Then you can install other packages by

```
pip install -r requirements.txt
mkdir submodules
cd submodules
git clone https://github.com/graphdeco-inria/diff-gaussian-rasterization.git --recursive
git clone https://gitlab.inria.fr/bkerbl/simple-knn.git
pip install ./diff-gaussian-rasterization/
pip install ./simple-knn/
cd ..
```

# Inference

To use the pre-trained model (provided or trained by yourself) to inference, you can choose one of the following methods. It needs about 16GB GPU VRAM.

### 1) Infer only one prompt

```
python inference.py --model_path /path/to/ckpt_file --prompt "input text prompt" --save_path /folder/to/save/videos --default_radius 3.5 --default_polar 60

# example
CUDA_VISIBLE_DEVICES=0 python inference.py --model_path models/vehicle.pth --prompt "Racing car, cyan, lightweight aero kit, sequential gearbox" --save_path workspace_inference --default_radius 3.5 --default_polar 60
```

### 2) Infer by user's input from command (without the requirements to load model at each time). You can input the prompt in command. Input "exit" to quit.

```
python inference_cmd.py --model_path /path/to/ckpt_file --save_path /folder/to/save/videos --default_radius 3.5 --default_polar 60

# example
CUDA_VISIBLE_DEVICES=0 python inference_cmd.py --model_path models/vehicle.pth --save_path workspace_inference --default_radius 3.5 --default_polar 60
```

### 3) GUI interactive method as previous demo shows.

\<optional\> If you want to start in a server and use in the local pc, you need construct a tunnel to server first.

```
ssh -L 5000:127.0.0.1:5000 <your server host>
```

Next, you can start the back-end program.

```
python inference_gui.py --model_path /path/to/ckpt_file

# example
CUDA_VISIBLE_DEVICES=0 python inference_gui.py --model_path models/vehicle.pth
```

Then you can open the page in your local browser.

```
127.0.0.1:5000
```

# Training

1) To accelerate training, we choose to cache the text embeddings in the training prompt set. But this may cost more RAM memory space and disk space. This method can save several minutes (speed up about 15%) for each epoch depending on the size of training prompts. You can also choose to mix the provided prompts into a single txt file for mixing training.

    ```
    python embedding_cache.py --prompts_set vehicle
    python embedding_cache.py --prompts_set daily_life
    python embedding_cache.py --prompts_set animal
    python embedding_cache.py --prompts_set mix
    ```

    The cached text embeddings will be saved at ./vehicle.pkl, ./daily_life.pkl, and ./animal.pkl.

2) Train the BrightDreamer generator. We provide the command demo of training in the following scripts.

    ```
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 sh scripts/vehicle.sh
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 sh scripts/daily_life.sh
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 sh scripts/animal.sh
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 sh scripts/mix_training.sh
    ```

    Key hyper-parameters:

    ```
    --prompts_set The training prompts set.

    --cache_file The cached text embeddings.

    --batch_size The number of prompts in a single iteration on each card. The actual batch size is the number of gpus * batch_size.

    --c_batch_size The number of cameras for a prompt to render images to calculate the SDS loss.

    --lr Learning rate.

    --eval_interval The frequency of outputing test images.

    --test_interval The frequency of rendering test videos.

    --guidance The Unet used to calculate the SDS loss.

    --workspace The folder of output.

    --ckpt Recover the training process.
    ```

    The batch_size of 8 and the c_batch_size of 4 may use 65GB GPU memory on a single card. In our experiments, 4 cards can also work well, but more slowly. Larger batch size will result in a better result. We train 36 hours for the vehicle prompts set, 60 hours for the daily life prompts set, and 30 hours for the animal prompts set on a server with 8 80GB GPUs.

# Potential Improvement Directions

* A better and more abundant prompts set will improve the training quality much more.
* The better diffusion model could improve our training quality.
* More training tricks can be introduced to our framework to improve the quality and to alleviate the 'Janus' promblem.


# Acknowledgement

Our code is inspired by [stable-dreamfusion](https://github.com/ashawkey/stable-dreamfusion), [Stable Diffusion](https://github.com/CompVis/stable-diffusion), [gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting) and [DeepFloyd-IF](https://huggingface.co/DeepFloyd/IF-I-XL-v1.0).
Thanks for their outstanding works and open-source!

# Citation

If you find this work useful, a citation will be appreciated via:

```
@misc{jiang2024brightdreamer,
    title={BrightDreamer: Generic 3D Gaussian Generative Framework for Fast Text-to-3D Synthesis}, 
    author={Lutao Jiang and Lin Wang},
    year={2024},
    eprint={2403.11273},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```
