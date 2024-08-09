Set up the repo following the instruction
update the torch to newest
They don't have a flag indicating the iter, so you need to change it in pixart_alpha_brecq.py
And I am unable to write the code for resume. so it is end to end image generation

command:
python pixart_alpha_brecq.py --plms --no_grad_ckpt --ddim_steps 50 --seed 40 --cond --wq 4 --ptq --aq 8 --outdir output --cali --skip_grid --use_aq --ckpt ./stable-diffusion/models/ldm/stable-diffusion-v1/sd-v1-4.ckpt --config stable-diffusion/configs/stable-diffusion/v1-inference.yaml --data_path /home/ruichen/data/annotations/captions_train2017.json --cali_save_path cali_save --pixart   