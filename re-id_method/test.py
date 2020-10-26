from tqdm import tqdm
from data.dataset import get_test_dataset_loader, get_query_dataset_loader
from models.re_id_model import get_model
from options.options import Options
import numpy as np
import torch
import torch.nn as nn
import os.path as osp
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader

if __name__ == "__main__":

    opt = Options(is_train = False).parse()

    test_data = get_test_dataset_loader(opt.dataroot) 
    test_dataloader = DataLoader(test_data, batch_size = opt.batch_size, shuffle = False)
    print('The number of testing images = %d' % len(test_data))
    print('The number of testing classes = %d' % test_data.total_ids)

    query_data = get_query_dataset_loader(opt.dataroot)
    dataloader_query = DataLoader(query_data, batch_size = 1, shuffle=False)
    print('The number of query images = %d' % len(query_data))
    print('The number of query classes = %d' % query_data.total_ids)


    model = get_model(1, opt.load_weights_path, opt.initial_suffix, opt.only_backbone, opt.device) 
    # print(model)

    # Computing gallery descriptors
    print('Computing gallery descriptors')
    backbone_model = model.get_backbone_model()
    backbone_model.eval()
    save_path = osp.join(opt.save_result_path, 'gallery_descriptors_{}.npy'.format(opt.initial_suffix))
    if osp.exists(save_path) and not opt.no_load:
        gallery = np.load(save_path, allow_pickle=True)
    else:
        gallery = []
        with torch.no_grad():
            for batch in tqdm(test_dataloader):
                batch['image'] = batch['image'].to(torch.device(opt.device))
                output = backbone_model(batch['image'])
                for i in range(batch['image'].size()[0]):
                    gallery.append([output[i].view(4096).cpu().numpy(), 
                                    batch['person_id'][i].cpu().numpy(), 
                                    batch['camera_id'][i].cpu().numpy()])
        gallery = np.asarray(gallery)
        np.save(save_path, gallery)

    # Computing cosine distances
    print('Computing cosine distanses')
    save_path = osp.join(opt.save_result_path, 'cos_dist_{}.npy'.format(opt.initial_suffix))
    if osp.exists(save_path) and not opt.no_load:
        cos_dist = np.load(save_path, allow_pickle=True)
    else:
        cos = nn.CosineSimilarity(dim=0, eps=1e-8)
        j = 0
        cos_dist = np.zeros((len(query_data), len(test_data) + 1, 3))
        with torch.no_grad():
            for i, batch in enumerate(tqdm(dataloader_query)):
                batch['image'] = batch['image'].to(torch.device(opt.device))
                output = backbone_model(batch['image'])
                output = output.view(1, 4096)
                
                for j, gallery_descriptor in enumerate(gallery):
                    cos_dist[i, j, 0], cos_dist[i, j, 1], cos_dist[i, j, 2] = gallery_descriptor[1], gallery_descriptor[2], cos(output[0].to(torch.device('cpu')), 
                                                                                                               torch.from_numpy(gallery_descriptor[0])).item()
                cos_dist[i, j + 1, 0] = batch['person_id'].item()
                cos_dist[i, j + 1, 1] = batch['camera_id'].item()
            
        cos_dist = np.asarray(cos_dist)
        np.save(save_path, cos_dist)

    # Computing rank-1
    print('Computing rank-1')
    correct = 0
    for element in cos_dist:
        person_id = element[-1, 0]
        camera_id = element[-1, 1]
        
        element = np.delete(element, np.where(element[:, 1] == camera_id), axis=0)
        
        if element[element[:,-1].argsort()][-1][0] == person_id:
            correct += 1

    rank_1 = correct / cos_dist.shape[0]

    # Computing mAP
    print('Computing mAP')
    tp_fn = []

    for element in cos_dist:
        person_id = element[-1, 0]
        camera_id = element[-1, 1]
        
        tp_fn.append(np.where(np.logical_and(element[:, 0] == person_id, element[:, 1] != camera_id))[0].shape[0])

    i = 0
    p_r_dots = []

    for element in cos_dist:
        person_id = element[-1, 0]
        camera_id = element[-1, 1]
        
        element = np.delete(element, np.where(element[:, 1] == camera_id), axis=0)
        element = element[element[:,-1].argsort()]
        
        tmp = np.zeros((element.shape[0], 2))
        tmp[np.argwhere(element[:, 0] == person_id), 0] = 1
        tmp[:, 1] = element[:, 2]
        p_r_dots.append(tmp)
        
    p_r_dots = np.asarray(p_r_dots)



    map_sum = 0
    for el in p_r_dots:
        y_true, y_score = el[:, 0], el[:, 1]
        map_sum += average_precision_score(y_true, y_score)
        
    mAP = map_sum / p_r_dots.shape[0]

    save_path = osp.join(opt.save_result_path, 'result_{}.txt'.format(opt.initial_suffix))
    with open(save_path, 'w') as result:
        result.write("Rank-1: {}\n".format(rank_1 * 100))
        print("Rank-1: ", rank_1 * 100)
        result.write("mAP: {}\n".format(mAP * 100))
        print("mAP: ", mAP * 100)