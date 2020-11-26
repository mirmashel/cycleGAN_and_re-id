# python orig_train.py --name orig_stylegan_pretrained_r1 --source_path ../../datasets/JTA_parsed/ --target_path ../../datasets/DukeMTMC/ \
# 	--n_mlp 8 \
# 	--sched --sample_iters 1000 --save_iters 10000 --multisize_vis --use_face_weights --loss r1 # --ckpt_name ./checkpoints/orig_stylegan/130000_64.model 

# python orig_train.py --name orig_stylegan_pretrained --source_path ../../datasets/JTA_parsed/ --target_path ../../datasets/DukeMTMC/ \
# 	--n_mlp 8 \
# 	--sched --sample_iters 1000 --save_iters 10000 --multisize_vis --use_face_weights --ckpt_name ./checkpoints/orig_stylegan_pretrained/110000_128.model 

# python orig_train.py --name orig_stylegan --source_path ../../datasets/JTA_parsed/ --target_path ../../datasets/DukeMTMC/ \
# 	--n_mlp 8 \
# 	--params_file ./params/big_iter_params --sched --sample_iters 1000 --save_iters 10000 --multisize_vis --ckpt_name ./checkpoints/orig_stylegan/130000_64.model 

# python train.py --name no_wgan_loss_v3 --source_path ../../datasets/JTA_parsed/ --target_path ../../datasets/DukeMTMC/ \
# 	--lambda_cls 1.5 --lambda_idt 1.5 --lambda_prcp 1.5 --no_use_gan --n_mlp 8 \
# 	--params_file ./params/default_iter_params --sched --sample_iters 1000 --save_iters 10000 --path_vgg_weights vgg_weights --prefix_vgg_weights imagenet #--ckpt_name ./checkpoints/first_attempt/000010_8.model

python train.py --name no_wgan_loss_v4 --source_path ../../datasets/JTA_parsed/ --target_path ../../datasets/DukeMTMC/ \
	--lambda_cls 1.5 --lambda_idt 1.5 --lambda_prcp 1.5 --lambda_msssim 1.5 --no_use_gan --n_mlp 8 \
	--params_file ./params/default_iter_params --sched --sample_iters 1000 --save_iters 10000 \
	--path_vgg_weights vgg_weights --prefix_vgg_weights imagenet --ckpt_name ./checkpoints/no_wgan_loss_v3/080000_64.model


# python train.py --name no_wgan_loss --source_path ../../datasets/JTA_parsed/ --target_path ../../datasets/DukeMTMC/ \
# 	--lambda_cls 1.5 --lambda_idt 5 --lambda_prcp 0 --no_use_gan \
# 	--params_file ./params/less_images_per_step --sched --sample_iters 500 --save_iters 10000 --path_vgg_weights vgg_weights --prefix_vgg_weights VGG_13 #--ckpt_name ./checkpoints/first_attempt/000010_8.model



# python train.py --name idt_5_prcp_10 --source_path ../../datasets/JTA_parsed/ --target_path ../../datasets/DukeMTMC/ \
# 	--lambda_cls 1.5 --lambda_idt 5 --lambda_prcp 10 \
# 	--sched --sample_iters 500 --save_iters 10000 --path_vgg_weights vgg_weights --prefix_vgg_weights VGG_13 #--ckpt_name ./checkpoints/first_attempt/000010_8.model
