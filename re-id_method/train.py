from tqdm import tqdm
from data.dataset import get_train_dataset_loader
from models.re_id_model import get_model
from models.scheduler import GradualWarmupScheduler
from options.options import Options
import torch
import torch.nn.functional as F

if __name__ == "__main__":

    opt = Options(is_train = True).parse()

    train_data = get_train_dataset_loader(opt.dataroot) 
    print('The number of training images = %d' % len(train_data))
    print('The number of training classes = %d' % train_data.total_ids)

    model = get_model(train_data.total_ids, opt.load_weights_path, opt.initial_suffix, opt.only_backbone, opt.device) 
    # print(model)


    optimizer = torch.optim.SGD(model.parameters(), lr = opt.lr) 
    scheduler_lr_red = torch.optim.lr_scheduler.StepLR(optimizer, opt.start_step_lr, 0.1) 
    scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier = 10, total_epoch = opt.warmup_epoch, after_scheduler = scheduler_lr_red) 


    try:
        for epoch in range(opt.start_epoch, opt.nepochs + 1 + opt.pretrain_classifiers_epochs):

            if epoch <= opt.pretrain_classifiers_epochs:
                optimizer.param_groups[0]['lr'] = opt.pretrain_classifiers_lr
            elif epoch == opt.pretrain_classifiers_epochs + 1:
                optimizer.param_groups[0]['lr'] = opt.lr

            print("Starts epoch ", epoch, ", Lr ", optimizer.param_groups[0]['lr'])
            
            loss_list = []
            acc_verification_list = []
            acc_first_id_list = []
            acc_second_id_list = []
            acc_first_id_val_list = [1]
            acc_second_id_val_list = [1]
            
            for i in tqdm(range(0, len(train_data), opt.batch_size)): 
                optimizer.zero_grad()
                
                images_first = torch.zeros((opt.batch_size, 3, 246, 128), dtype = torch.float32, device = opt.device)
                images_second = torch.zeros((opt.batch_size, 3, 246, 128), dtype = torch.float32, device = opt.device)
                persons_id_first = torch.zeros((opt.batch_size, ), dtype = torch.long, device = opt.device)
                persons_id_second = torch.zeros((opt.batch_size, ), dtype = torch.long, device = opt.device)
                verification = torch.zeros(opt.batch_size, dtype = torch.long, device = opt.device)
                
                for j, k in enumerate(range(i, i + opt.batch_size)):
                    k = torch.randint(0, len(train_data), (1,)).item()
                    first_image = train_data[k]
                    
                    images_first[j] = first_image["image"]
                    persons_id_first[j] = first_image["person_id"]
                    
                    r = torch.rand((1, )).item()

                    if r <= 0.4:
                        verification[j] = 1
                        idx = train_data.get_random_class_image_idx(persons_id_first[j].item(), True)
                    else:
                        idx = train_data.get_random_class_image_idx(persons_id_first[j].item(), False)
                        
                    second_image = train_data[idx]
                    images_second[j] = second_image["image"]
                    persons_id_second[j] = second_image["person_id"]
                    

                output = model(images_first, images_second, epoch <= opt.pretrain_classifiers_epochs)

                loss_verification = F.cross_entropy(output[0], verification)
                loss_first_id = F.cross_entropy(output[1], persons_id_first)
                loss_second_id = F.cross_entropy(output[2], persons_id_second)
                
                loss = loss_verification + 0.5 * (loss_first_id + loss_second_id)

                loss.backward()
                optimizer.step()

                loss_list.append(loss)
                
                predicted = torch.argmax(output[0], 1)
                correct = (predicted == verification).sum().item()
                acc_verification_list.append(correct / verification.size()[0])
                
                predicted = torch.argmax(output[1], 1)
                correct = (predicted == persons_id_first).sum().item()
                acc_first_id_list.append(correct / persons_id_first.size()[0])
                
                predicted = torch.argmax(output[2], 1)
                correct = (predicted == persons_id_second).sum().item()
                acc_second_id_list.append(correct / persons_id_second.size()[0])
                
            if epoch > opt.pretrain_classifiers_epochs:
                scheduler_warmup.step()
                
            if epoch % opt.checkpoint_every == 0 and epoch != 0: 
                model.save_model(opt.save_weights_path, epoch) 

            print('{}\'s epoch is over, Loss: {:.4f}, Verification Accuracy: {:.6f}%, First Person ID Accuracy: {:.6f}%, Second Person ID Accuracy: {:.6f}%, First Person ID Val Accuracy: {:.6f}%, Second Person ID Val Accuracy: {:.6f}%\n'
                  .format(epoch, sum(loss_list) / float(len(loss_list)), 
                          sum(acc_verification_list) / float(len(acc_verification_list)), 
                          sum(acc_first_id_list) / float(len(acc_first_id_list)),
                          sum(acc_second_id_list) / float(len(acc_second_id_list)),
                          sum(acc_first_id_val_list) / float(len(acc_first_id_val_list)),
                          sum(acc_second_id_val_list) / float(len(acc_second_id_val_list))))
    except KeyboardInterrupt:
        pass

    model.save_model(opt.save_weights_path, "latest" if opt.save_suffix == '' else opt.save_suffix)