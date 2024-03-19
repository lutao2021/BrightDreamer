# lr 5e-5 may make training instable, you can attempt to adjust it
python main.py --workspace workspace_daily \
--iters 600000 --lr 5e-5 --guidance IF --h 512 --w 512 \
--batch_size 8 --c_batch_size 4 --radius_range 2.5 2.5 --default_radius 2.5 --eval_interval 1 --save_interval 5 \
--port 12369 --fp16 --grad_clip 5.0 --xyzres --perpneg --test_interval 20 --ema_decay 0.99 \
--prompts_set daily_life --cache_path ./daily_life_if.pkl