python train.py --name first_attempt --source_path ../../datasets/JTA_parsed/ --target_path ../../datasets/DukeMTMC/ \
	--lambda_cls 1.5 --lambda_idt 1.5 --lambda_prcp 2.5 \
	--sched --phase 10000 --sample_iters 500 --save_iters 10000 --path_vgg_weights vgg_weights --prefix_vgg_weights VGG_13 #--ckpt_name ./checkpoints/first_attempt/000010_8.model
