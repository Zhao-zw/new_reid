from tri_loss.dataset import create_dataset
import torch
import numpy as np
from torch.autograd import Variable
from tri_loss.utils.utils import set_devices
import pdb

def Get_data(cfg):

    if not cfg.only_test:
        train_set = create_dataset(**cfg.train_set_kwargs)
        # The combined dataset does not provide val set currently.
        val_set = None if cfg.dataset == 'combined' else create_dataset(**cfg.val_set_kwargs)
        return train_set,val_set
    test_sets = []
    test_set_names = []
    if cfg.dataset == 'combined':
        for name in ['market1501', 'cuhk03', 'duke']:
            cfg.test_set_kwargs['name'] = name
            test_sets.append(create_dataset(**cfg.test_set_kwargs))
            test_set_names.append(name)
        return test_sets,test_set_names
    else:
        test_sets.append(create_dataset(**cfg.test_set_kwargs))
        test_set_names.append(cfg.dataset)
        return test_sets,test_set_names
    
def lambda_rule(epoch):
    lr_l = 1.0 - max(0, epoch + 2 -100 ) / float(100 + 1)
    return lr_l

def scheduler(a,b,c,d):
    scheduler_Di = torch.optim.lr_scheduler.LambdaLR(a, lr_lambda=lambda_rule)
    scheduler_Dp = torch.optim.lr_scheduler.LambdaLR(b, lr_lambda=lambda_rule)
    scheduler_Dt = torch.optim.lr_scheduler.LambdaLR(c, lr_lambda=lambda_rule)
    scheduler_G = torch.optim.lr_scheduler.LambdaLR(d, lr_lambda=lambda_rule)
    return scheduler_Di,scheduler_Dp,scheduler_Dt,scheduler_G

def get_the_pn_truth_pose_nosie(p_inds,n_inds,ims,noise,pose):
    'get the batch  positive truth picture '
    p_picture = []
    for p_ind in p_inds:
        p_picture.append(ims[p_ind])
    p_picture = np.array(p_picture)

    'get the batch positive truth picture_nosie'
    p_nosie = []
    for p_ind in p_inds:
        p_nosie.append(noise.numpy()[p_ind])
    p_nosie = np.array(p_nosie)

    'get the batch positive truth picture_pose'
    p_pose = []
    for p_ind in p_inds:
        p_pose.append(pose[p_ind])
    p_pose = np.array(p_pose)
    'get the batch negative picture'
    n_picture = []
    for n_ind in n_inds:
        n_picture.append(ims[n_ind])
    n_picture = np.array(n_picture)

    'get the batch negative truth picture_nosie'
    n_nosie = []
    for n_ind in n_inds:
        n_nosie.append(noise.numpy()[n_ind])
    n_nosie = np.array(n_nosie)

    'get the batch positive truth picture_pose'
    n_pose = []
    for n_ind in n_inds:
        n_pose.append(pose[n_ind])
    n_pose = np.array(n_pose)
    return p_picture,p_nosie,p_pose,n_picture,n_nosie,n_pose

def get_the_FDGAN_input(cfg,labels,ims,pose,pose_target,noise):
    'get the GAN input label'
    TVT, TMO = set_devices(cfg.sys_device_ids)

    data_random_state = np.random.get_state()
    input1_label = labels.copy()
    input2_label = labels.copy()
    np.random.shuffle(input2_label)
    labels_v = (torch.from_numpy(input1_label).long() == torch.from_numpy(input2_label).long()).long()
    labels_v_var = TVT(labels_v.long())
      

    'get the GAN origin picture'
    input1_ims = ims.copy()   
    input2_ims = ims.copy()
    np.random.set_state(data_random_state)
    np.random.shuffle(input2_ims)
    origin = torch.cat([torch.from_numpy(input1_ims).float(),torch.from_numpy(input2_ims).float()])
    origin_var = Variable(TVT(origin))
    bs = origin_var.size(0)

    'get the GAN pose'
    input1_map = pose.copy()
    input2_map = pose.copy()  
    np.random.set_state(data_random_state)
    np.random.shuffle(input2_map)
    mask = labels_v.view(-1,1,1,1).expand_as(torch.from_numpy(input1_map))
    input2_map = torch.from_numpy(input1_map).float()*mask.float() + torch.from_numpy(input2_map).float()*(1-mask.float())
    B_map = torch.cat([torch.from_numpy(input1_map).float(),input2_map])
    B_map_var = Variable(TVT(B_map.float()))

    'get the GAN target'
    target1 = pose_target.copy()
    target2 = pose_target.copy()
    np.random.set_state(data_random_state)
    np.random.shuffle(target2)
    mask = labels_v.view(-1,1,1,1).expand_as(torch.from_numpy(target1))
    target2 = torch.from_numpy(target1).float()*mask.float() + torch.from_numpy(target2).float()*(1-mask.float())
    GAN_target = torch.cat([torch.from_numpy(target1).float(),target2])
    GAN_target_var = Variable(TVT(GAN_target.float()))
      #pdb.set_trace()


    noisess = torch.cat((noise,noise))
    z = Variable(TVT(noisess.float()))

    return labels_v_var,origin_var,bs,B_map_var,GAN_target_var,z