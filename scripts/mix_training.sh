python main.py --workspace workspace_mix \
--iters 600000 --lr 2e-5 --guidance IF --h 512 --w 512 \
--batch_size 8 --c_batch_size 4 --radius_range 2.5 2.5 --default_radius 2.5 --eval_interval 1 --save_interval 5 \
--port 12369 --fp16 --grad_clip 5.0 --xyzres --perpneg --side_decay_factor 5.0 --negative_w -3.0 --test_interval 20 --ema_decay 0.99 \
--prompts_set mix
