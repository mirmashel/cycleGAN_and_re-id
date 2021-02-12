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

# python train.py --name no_wgan_loss_v4 --source_path ../../datasets/JTA_parsed/ --target_path ../../datasets/DukeMTMC/ \
# 	--lambda_cls 1.5 --lambda_idt 1.5 --lambda_prcp 1.5 --lambda_msssim 1.5 --no_use_gan --n_mlp 8 \
# 	--params_file ./params/default_iter_params --sched --sample_iters 1000 --save_iters 10000 \
# 	--path_vgg_weights vgg_weights --prefix_vgg_weights imagenet --ckpt_name ./checkpoints/no_wgan_loss_v3/110000_128.model

# python train.py --name no_wgan_loss_v6 --source_path ../../datasets/JTA_parsed/ --target_path ../../datasets/DukeMTMC/ \
# 	--lambda_cls 1.5 --lambda_idt 2.5 --lambda_prcp 2.5 --lambda_msssim 1.5 --no_use_gan --n_mlp 4 --static_noise \
# 	--params_file ./params/default_iter_params --sched --sample_iters 1000 --save_iters 10000 \
# 	--path_vgg_weights vgg_weights --prefix_vgg_weights imagenet # --ckpt_name ./checkpoints/no_wgan_loss_v3/110000_128.model

# python train.py --name no_wgan_loss_v7 --source_path ../../datasets/JTA_parsed/ --target_path ../../datasets/DukeMTMC/ \
# 	--lambda_cls 1.5 --lambda_idt 2.5 --lambda_prcp 2.5 --lambda_msssim 1.5 --no_use_gan --n_mlp 4 --static_noise --active_styles 10 \
# 	--params_file ./params/default_iter_params --sched --sample_iters 1000 --save_iters 10000 \
# 	--path_vgg_weights vgg_weights --prefix_vgg_weights imagenet --ckpt_name ./checkpoints/no_wgan_loss_v6/080000_64.model

# python train.py --name no_wgan_loss_v8 --source_path ../../datasets/JTA_parsed/ --target_path ../../datasets/DukeMTMC/ \
# 	--lambda_cls 0 --lambda_idt 2.5 --lambda_prcp 2.5 --lambda_msssim 1.5 --no_use_gan --n_mlp 4 --decoder resnet50 --static_noise --code_size 2048 \
# 	--params_file ./params/default_iter_params --sched --sample_iters 1000 --save_iters 10000 \
# 	--path_vgg_weights vgg_weights --prefix_vgg_weights imagenet # --ckpt_name ./checkpoints/no_wgan_loss_v6/080000_64.model

# python train.py --name no_wgan_loss_v9 --source_path ../../datasets/JTA_parsed/ --target_path ../../datasets/DukeMTMC/ \
# 	--lambda_cls 1.5 --lambda_idt 2.5 --lambda_prcp 2.5 --lambda_msssim 1.5 --no_use_gan --n_mlp 4 --decoder resnet50 --static_noise --code_size 1048 \
# 	--params_file ./params/default_iter_params --sched --sample_iters 1000 --save_iters 10000 \
# 	--path_vgg_weights vgg_weights --prefix_vgg_weights imagenet # --ckpt_name ./checkpoints/no_wgan_loss_v6/080000_64.model

# python train.py --name no_wgan_loss_v11 --source_path ../../datasets/JTA_parsed/ --target_path ../../datasets/DukeMTMC/ \
	# --lambda_cls 1.5 --lambda_idt 2.5 --lambda_prcp 2.5 --lambda_msssim 1.5 --no_use_gan --n_mlp 4 --decoder base --static_noise \
	# --params_file ./params/default_iter_params --sched --sample_iters 1000 --save_iters 10000 \
	# --path_vgg_weights vgg_weights --prefix_vgg_weights imagenet # --ckpt_name ./checkpoints/no_wgan_loss_v6/080000_64.model

# python train.py --name no_wgan_loss_v12 --source_path ../../datasets/JTA_parsed/ --target_path ../../datasets/DukeMTMC/ \
# 	--lambda_cls 1.5 --lambda_idt 2.5 --lambda_prcp 2.5 --lambda_msssim 1.5 --no_use_gan --n_mlp 4 --decoder resnet50 --static_noise \
# 	--params_file ./params/default_iter_params --sched --sample_iters 1000 --save_iters 10000 \
# 	--path_vgg_weights vgg_weights --prefix_vgg_weights imagenet # --ckpt_name ./checkpoints/no_wgan_loss_v6/080000_64.model

# python train.py --name no_wgan_loss --source_path ../../datasets/JTA_parsed/ --target_path ../../datasets/DukeMTMC/ \
# 	--lambda_cls 1.5 --lambda_idt 5 --lambda_prcp 0 --no_use_gan \
# 	--params_file ./params/less_images_per_step --sched --sample_iters 500 --save_iters 10000 --path_vgg_weights vgg_weights --prefix_vgg_weights VGG_13 #--ckpt_name ./checkpoints/first_attempt/000010_8.model

# python train.py --name idt_5_prcp_10 --source_path ../../datasets/JTA_parsed/ --target_path ../../datasets/DukeMTMC/ \
# 	--lambda_cls 1.5 --lambda_idt 5 --lambda_prcp 10 \
# 	--sched --sample_iters 500 --save_iters 10000 --path_vgg_weights vgg_weights --prefix_vgg_weights VGG_13 #--ckpt_name ./checkpoints/first_attempt/000010_8.model



# python train.py --name encoder_decoder_v1 --source_path ../../datasets/JTA_parsed/ --target_path ../../datasets/DukeMTMC/ \
# 	--lambda_cls 0 --lambda_idt 1 --lambda_prcp 0 --lambda_msssim 0 --no_use_gan --n_mlp 8 --decoder base --static_noise --code_first \
# 	--params_file ./params/default_iter_params --sched --sample_iters 1000 --save_iters 10000 \
# 	--path_vgg_weights vgg_weights --prefix_vgg_weights imagenet # --ckpt_name ./checkpoints/no_wgan_loss_v6/080000_64.model

# python train.py --name encoder_decoder_v2 --source_path ../../datasets/JTA_parsed/ --target_path ../../datasets/DukeMTMC/ \
# 	--lambda_cls 1 --lambda_idt 1 --lambda_prcp 0 --lambda_msssim 0 --no_use_gan --n_mlp 8 --decoder base --static_noise --code_first \
# 	--params_file ./params/default_iter_params --sched --sample_iters 1000 --save_iters 10000 \
# 	--path_vgg_weights vgg_weights --prefix_vgg_weights imagenet # --ckpt_name ./checkpoints/no_wgan_loss_v6/080000_64.model

# python train.py --name encoder_decoder_v3 --source_path ../../datasets/JTA_parsed/ --target_path ../../datasets/DukeMTMC/ \
# 	--lambda_cls 0 --lambda_idt 1 --lambda_prcp 1 --lambda_msssim 0 --no_use_gan --n_mlp 8 --decoder base --static_noise --code_first \
# 	--params_file ./params/default_iter_params --sched --sample_iters 1000 --save_iters 10000 \
# 	--path_vgg_weights vgg_weights --prefix_vgg_weights imagenet # --ckpt_name ./checkpoints/no_wgan_loss_v6/080000_64.model

# python train.py --name encoder_decoder_v4 --source_path ../../datasets/JTA_parsed/ --target_path ../../datasets/DukeMTMC/ \
# 	--lambda_cls 0 --lambda_idt 1 --lambda_prcp 1 --lambda_msssim 1 --no_use_gan --n_mlp 8 --decoder base --static_noise --code_first \
# 	--params_file ./params/default_iter_params --sched --sample_iters 1000 --save_iters 10000 \
# 	--path_vgg_weights vgg_weights --prefix_vgg_weights imagenet --ckpt_name ./checkpoints/encoder_decoder_v3/080000_64.model

# python train.py --name encoder_decoder_v5 --source_path ../../datasets/JTA_parsed/ --target_path ../../datasets/DukeMTMC/ \
# 	--lambda_cls 1 --lambda_idt 1 --lambda_prcp 1 --lambda_msssim 0 --no_use_gan --n_mlp 8 --decoder base --static_noise --code_first \
# 	--params_file ./params/default_iter_params --sched --sample_iters 1000 --save_iters 10000 \
# 	--path_vgg_weights vgg_weights --prefix_vgg_weights imagenet --ckpt_name ./checkpoints/encoder_decoder_v5/150000_128.model

# python train.py --name encoder_decoder_v6 --source_path ../../datasets/JTA_parsed/ --target_path ../../datasets/DukeMTMC/ \
# 	--lambda_cls 0.4 --lambda_idt 1.5 --lambda_prcp 1 --lambda_msssim 0 --no_use_gan --n_mlp 8 --decoder base --static_noise --code_first \
# 	--params_file ./params/default_iter_params --sched --sample_iters 1000 --save_iters 10000 \
# 	--path_vgg_weights vgg_weights --prefix_vgg_weights imagenet # --ckpt_name ./checkpoints/no_wgan_loss_v6/080000_64.model

# python train.py --name encoder_decoder_v7 --source_path ../../datasets/JTA_parsed/ --target_path ../../datasets/DukeMTMC/ \
# 	--lambda_cls 0.4 --lambda_idt 1 --lambda_prcp 1.5 --lambda_msssim 0 --no_use_gan --n_mlp 8 --decoder base --static_noise --code_first \
# 	--params_file ./params/default_iter_params --sched --sample_iters 1000 --save_iters 10000 \
# 	--path_vgg_weights vgg_weights --prefix_vgg_weights imagenet # --ckpt_name ./checkpoints/no_wgan_loss_v6/080000_64.model

# попробовать сильно больше итераций с лучшим из 6 или 7
# python train.py --name encoder_decoder_v8 --source_path ../../datasets/JTA_parsed/ --target_path ../../datasets/DukeMTMC/ \
# 	--lambda_cls 0.4 --lambda_idt 1 --lambda_prcp 1.5 --lambda_msssim 0 --no_use_gan --n_mlp 8 --decoder base --static_noise --code_first \
# 	--params_file ./params/big_iter_params --sched --sample_iters 1000 --save_iters 10000 \
# 	--path_vgg_weights vgg_weights --prefix_vgg_weights imagenet # --ckpt_name ./checkpoints/no_wgan_loss_v6/080000_64.model

# попробовать предобученный на лицах с параметрами из 6 или 7 
# python train.py --name encoder_decoder_v9 --source_path ../../datasets/JTA_parsed/ --target_path ../../datasets/DukeMTMC/ \
# 	--lambda_cls 0.4 --lambda_idt 1 --lambda_prcp 1.5 --lambda_msssim 0 --no_use_gan --n_mlp 8 --decoder base --static_noise --code_first --use_face_weights \
# 	--params_file ./params/default_iter_params --sched --sample_iters 1000 --save_iters 10000 \
# 	--path_vgg_weights vgg_weights --prefix_vgg_weights imagenet # --ckpt_name ./checkpoints/no_wgan_loss_v6/080000_64.model


python train.py --name encoder_decoder_v8_disc --source_path ../../datasets/JTA_parsed/ --target_path ../../datasets/DukeMTMC/ \
	--lambda_cls 0.4 --lambda_idt 1 --lambda_prcp 1.5 --lambda_msssim 0 --n_mlp 8 --decoder base --static_noise --code_first \
	--params_file ./params/big_iter_params --sched --sample_iters 1000 --save_iters 10000 \
	--path_vgg_weights vgg_weights --prefix_vgg_weights imagenet # --ckpt_name ./checkpoints/no_wgan_loss_v6/080000_64.model