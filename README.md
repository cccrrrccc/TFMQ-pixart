Set up the repo following the instruction
update the torch to newest
They don't have a flag indicating the iter, so you need to change it in pixart_alpha_brecq.py
And I am unable to write the code for resume. so it is end to end image generation

Install environment:

(1)
conda env create -f ./stable-diffusion/environment.yaml
conda activate ldm
pip install -r requirements.txt

(2)
Install pytorch 2.4.0 with cuda (here is a command I paste from their official site)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

And install diffusers 0.29.2 (for what we are using right now)

(3)
Copy the captions folders we used to this working directory

(4)
Change pixart_alpha_brecq.py L431 to fit your path

(5) (Optional)
And the other thing, is the _execution_device problem. Probably you need to edit the diffusers source code.

(6)
The number of iterations is hardcode in the pixart_alpha_brecq.py. Please change it in the code.

command:
Alpha:
I remove some unuseful flag. But if the first command doesn't work, try the second one.

python pixart_alpha_brecq.py --plms --no_grad_ckpt --ddim_steps 50 --seed 40 --cond --wq 4 --ptq --aq 8 --outdir output --cali --skip_grid --use_aq --data_path /home/ruichen/data/annotations/captions_train2017.json --cali_save_path cali_save --pixart   

python pixart_alpha_brecq.py --plms --no_grad_ckpt --ddim_steps 50 --seed 40 --cond --wq 4 --ptq --aq 8 --outdir output --cali--skip_grid --use_aq --ckpt ./stable-diffusion/models/ldm/stable-diffusion-v1/sd-v1-4.ckpt --config stable-diffusion/configsstable-diffusion/v1-inference.yaml --data_path /home/ruichen/data/annotations/captions_train2017.json --cali_save_path cali_save --pixart

Sigma:
It is the same with alpha except the file name
python pixart_sigma_brecq.py --plms --no_grad_ckpt --ddim_steps 50 --seed 40 --cond --wq 4 --ptq --aq 8 --outdir output --cali --skip_grid --use_aq --data_path /home/ruichen/data/annotations/captions_train2017.json --cali_save_path cali_save --pixart   