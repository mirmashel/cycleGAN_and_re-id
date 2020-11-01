# python train.py --dataroot ./datasets/jta_2_duke/ --name exp_resnet9 --model cycle_gan --netG resnet_9blocks --n_epochs 10 --n_epochs_decay 10

# python train.py --dataroot ./datasets/jta_2_duke/ --name exp_resnet6 --model cycle_gan --netG resnet_6blocks --n_epochs 10 --n_epochs_decay 10

# python generate_results.py --dataroot datasets/jta_2_duke/ --name exp_resnet9 --model cycle_gan --num_test 200000 --netG resnet_9blocks --results_dir ../../datasets/jta_2_duke_resnet9/bounding_box_train/ --load_size 256  

# python generate_results.py --dataroot datasets/jta_2_duke/ --name exp_resnet6 --model cycle_gan --num_test 200000 --netG resnet_6blocks --results_dir ../../datasets/jta_2_duke_resnet6/bounding_box_train/ --load_size 256  

python train.py --dataroot ./datasets/jta_2_duke/ --name exp_resnet9_sp --model cycle_gan --netG resnet_9blocks --n_epochs 10 --n_epochs_decay 10 --use_SP

python generate_results.py --dataroot datasets/jta_2_duke/ --name exp_resnet9_sp --model cycle_gan --num_test 200000 --netG resnet_9blocks --results_dir ../../datasets/jta_2_duke_resnet9_sp/bounding_box_train/ --load_size 256