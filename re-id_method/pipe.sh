python train.py --name DukeMTMC --nepochs 160 --warmup_epoch 6 --start_step_lr 40 --checkpoint_every 80 --dataroot ../../datasets/DukeMTMC/ --gpu_id 0

python test.py --name DukeMTMC --dataroot ../../datasets/DukeMTMC/ --gpu_id 0

# python train.py --name DukeMTMC__JTA_parsed_same_epochs --nepochs 160 --warmup_epoch 6 --start_step_lr 40 --checkpoint_every 80 --dataroot ../../datasets/DukeMTMC/ --dataroot ../../datasets/JTA_parsed/ --gpu_id 0

# python test.py --name DukeMTMC__JTA_parsed_same_epochs --dataroot ../../datasets/DukeMTMC/ --gpu_id 0

python train.py --name DukeMTMC__JTA_parsed --nepochs 80 --warmup_epoch 3 --start_step_lr 20 --checkpoint_every 40 --dataroot ../../datasets/DukeMTMC/ --dataroot ../../datasets/JTA_parsed/ --gpu_id 0

python test.py --name DukeMTMC__JTA_parsed --dataroot ../../datasets/DukeMTMC/ --gpu_id 0

python train.py --name DukeMTMC__jta_2_duke_resnet9 --nepochs 80 --warmup_epoch 3 --start_step_lr 20 --checkpoint_every 40 --dataroot ../../datasets/DukeMTMC/ --dataroot ../../datasets/jta_2_duke_resnet9/ --gpu_id 0

python test.py --name DukeMTMC__jta_2_duke_resnet9 --dataroot ../../datasets/DukeMTMC/ --gpu_id 0

python train.py --name JTA_parsed --nepochs 160 --warmup_epoch 6 --start_step_lr 40 --checkpoint_every 80 --dataroot ../../datasets/JTA_parsed/ --gpu_id 0

python test.py --name JTA_parsed --dataroot ../../datasets/DukeMTMC/ --gpu_id 0

python train.py --name jta_2_duke_resnet9 --nepochs 160 --warmup_epoch 6 --start_step_lr 40 --checkpoint_every 80 --dataroot ../../datasets/jta_2_duke_resnet9/ --gpu_id 0

python test.py --name jta_2_duke_resnet9 --dataroot ../../datasets/DukeMTMC/ --gpu_id 0
