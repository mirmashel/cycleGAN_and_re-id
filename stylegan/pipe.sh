python train.py --name idt_15_prcp_20 --source_path ../../datasets/JTA_parsed/ --target_path ../../datasets/DukeMTMC/ \
	--lambda_cls 1.5 --lambda_idt 15 --lambda_prcp 20 \
	--sched --phase 10000 --sample_iters 500 --save_iters 10000 --path_vgg_weights vgg_weights --prefix_vgg_weights VGG_13 #--ckpt_name ./checkpoints/first_attempt/000010_8.model
# big_idt_prcp - 10, 10