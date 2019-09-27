from __future__ import print_function

import sys
import os.path as osp 
sys.path.insert(0, '.')
import numpy as np
import torch
from torch.autograd import Variable
import torch.optim as optim
from torch.nn.parallel import DataParallel
from torch.nn import functional as F

import random

import time
from tensorboardX import SummaryWriter



from FDGAN.utils.visualizer import Visualizer
from FDGAN.model import create
from FDGAN.model.embedding import EltwiseSubEmbed
from FDGAN.model.multi_branch import SiameseNet

from script.experiment.Config import Config,ExtractFeature

from tri_loss.dataset import create_dataset
from tri_loss.model.Model import Model
from tri_loss.model.TripletLoss import TripletLoss
from tri_loss.model.loss import global_loss,normalize
from FDGAN.model.PoseGenerator import get_norm_layer, set_bn_fix,CustomPoseGenerator, init_weights,NLayerDiscriminator,_load_state_dict,get_current_visuals,save_network
from torch.nn import functional as F


from tri_loss.utils.utils import tight_float_str as tfs
from tri_loss.utils.utils import may_set_mode
from tri_loss.utils.utils import load_state_dict
from tri_loss.utils.utils import load_ckpt
from tri_loss.utils.utils import save_ckpt
from tri_loss.utils.utils import set_devices
from tri_loss.utils.utils import AverageMeter
from tri_loss.utils.utils import to_scalar
from tri_loss.utils.utils import ReDirectSTD
from tri_loss.utils.utils import set_seed
from tri_loss.utils.utils import adjust_lr_exp
from tri_loss.utils.utils import adjust_lr_staircase


from tri_loss_gan.dataset import Get_data,scheduler,get_the_pn_truth_pose_nosie,get_the_FDGAN_input
from FDGAN.model.losses import GANLoss
import pdb



def main():
  cfg = Config()

  # Redirect logs to both console and file.
  if cfg.log_to_file:
    ReDirectSTD(cfg.stdout_file, 'stdout', False)
    ReDirectSTD(cfg.stderr_file, 'stderr', False)

  # Lazily create SummaryWriter
  writer = None

  TVT, TMO = set_devices(cfg.sys_device_ids)

  if cfg.seed is not None:
    set_seed(cfg.seed)

  # Dump the configurations to log.
  import pprint
  print('-' * 60)
  print('cfg.__dict__')
  pprint.pprint(cfg.__dict__)
  print('-' * 60)

  ###########
  # Dataset #
  ###########
  if not cfg.only_test:
    train_set,val_set = Get_data(cfg)
  else :
    test_sets,test_set_names = Get_data(cfg)



  ###########
  # Models  #
  ###########

  #visualizer = Visualizer(cfg)

  model = Model(last_conv_stride=cfg.last_conv_stride)
  model_net_G = CustomPoseGenerator(cfg.pose_feature_size, 2048, cfg.noise_feature_size,
                                dropout=cfg.drop, norm_layer=get_norm_layer(norm_type=cfg.norm), fuse_mode=cfg.fuse_mode, connect_layers=cfg.connect_layers)

  e_base_model = create('resnet50', cut_at_pooling=True)
  e_embed_model = EltwiseSubEmbed(use_batch_norm=True, use_classifier=True, num_features=2048, num_classes=2)
  model_net_E = SiameseNet(e_base_model, e_embed_model)


  di_base_model = create('resnet50', cut_at_pooling=True)
  di_embed_model = EltwiseSubEmbed(use_batch_norm=True, use_classifier=True, num_features=2048, num_classes=1)
  model_net_Di = SiameseNet(di_base_model, di_embed_model)


  model_net_Dp = NLayerDiscriminator(3+18, norm_layer=get_norm_layer(norm_type=cfg.norm))

  model_net_Dt = NLayerDiscriminator(1+3+3, norm_layer=get_norm_layer(norm_type=cfg.norm)).cuda()
  #print(dg_base_model)
  #pdb.set_trace()
  #pdb.set_trace()
  #init_weights(model_net_Di)
  #init_weights(model_net_Dp)
  #init_weights(model_net_G)
  #init_weights(model_net_E)
  # Model wrapper

  model_w = DataParallel(model).cuda()
  model_net_G_w =  DataParallel(model_net_G).cuda()
  model_net_E_w = DataParallel(model_net_E).cuda()
  model_net_Di_w =  DataParallel(model_net_Di).cuda()
  model_net_Dp_w =  DataParallel(model_net_Dp).cuda()
  model_net_Dt_w =  DataParallel(model_net_Dt).cuda()


  #############################
  # Criteria and Optimizers   #
  #############################

  GAN_D_loss = GANLoss(smooth=True).cuda()
  GAN_G_loss = GANLoss(smooth=False).cuda()
  rand_list = [True] * 1 + [False] * 10000



  tri_loss = TripletLoss(margin=cfg.margin)
  optimizer = optim.Adam(model.parameters(),
                         lr=cfg.base_lr,
                         weight_decay=cfg.weight_decay)


  modules_optims = [model, optimizer]



  param_groups = [{'params': model_net_E.base_model.parameters(), 'lr_mult': 0.1},
                 {'params': model_net_E.embed_model.parameters(), 'lr_mult': 1.0},
                  {'params': model_net_G.parameters(), 'lr_mult': 0.1}]
  optimizer_G = torch.optim.Adam(param_groups,
                                  lr=0.001*0.1, betas=(0.5, 0.999))
  optimizer_Di = torch.optim.SGD(model_net_Di.parameters(),
                                          lr=0.001, momentum=0.9, weight_decay=1e-4)
  optimizer_Dp = torch.optim.SGD(model_net_Dp.parameters(),
                                                lr=0.001, momentum=0.9, weight_decay=1e-4)
  optimizer_Dt = torch.optim.SGD(model_net_Dt.parameters(),
                                                lr=0.001, momentum=0.9, weight_decay=1e-4)
  all_model_optims = [model,model_net_G,model_net_E,model_net_Di,model_net_Dp,model_net_Dt,optimizer,optimizer_G,optimizer_Di,optimizer_Dp,optimizer_Dt]
  all_optims = [optimizer_G,optimizer_Dt,optimizer_Dp,optimizer_Di]
  ################################
  # May Resume Models and Optims #
  ################################

  if cfg.resume:
    resume_ep, scores = load_ckpt(modules_optims, cfg.ckpt_file)
    _, _ = load_ckpt(all_optims, cfg.user_dir+'new_reid/Result/tlg/optim.pth')
    _load_state_dict(model_net_G,cfg.user_dir+'new_reid/Result/tlg/'+str(resume_ep)+'_net_G.pth')
    _load_state_dict(model_net_Di,cfg.user_dir+'new_reid/Result/tlg/'+str(resume_ep)+'_net_Di.pth')
    _load_state_dict(model_net_E,cfg.user_dir+'new_reid/Result/tlg/'+str(resume_ep)+'_net_E.pth')
    _load_state_dict(model_net_Dp,cfg.user_dir+'new_reid/Result/tlg/'+str(resume_ep)+'_net_Dp.pth')
  

  if cfg.use_FDGAN_model:
    _load_state_dict(model_net_G,cfg.user_dir+'new_reid/Result/tlg/FD/best_net_G.pth')
    _load_state_dict(model_net_E,cfg.user_dir+'new_reid/Result/tlg/FD/best_net_E.pth')
    _load_state_dict(model_net_Di,cfg.user_dir+'new_reid/Result/tlg/FD/best_net_Di.pth')
    _load_state_dict(model_net_Dp,cfg.user_dir+'new_reid/Result/tlg/FD/best_net_Dp.pth')


  if cfg.resume == False & cfg.use_FDGAN_model == False:  
    init_weights(model_net_Di)
    init_weights(model_net_Dp)
    init_weights(model_net_Dt)
    init_weights(model_net_G)
    init_weights(model_net_E)
  

  # May Transfer Models and Optims to Specified Device. Transferring optimizer
  # is to cope with the case when you load the checkpoint to a new device.
  TMO(all_model_optims)

  ########
  # Test #
  ########

  def test(load_model_weight=False):
    if load_model_weight:
      if cfg.model_weight_file != '':
        map_location = (lambda storage, loc: storage)
        ckpt = torch.load(cfg.model_weight_file, map_location=map_location)
        for m, sd in zip(modules_optims, ckpt['state_dicts']):

          m.load_state_dict(sd)
        print('Loaded model weights from {}'.format(cfg.model_weight_file))
      else:
        load_ckpt(modules_optims, cfg.ckpt_file)

    for test_set, name in zip(test_sets, test_set_names):
      test_set.set_feat_func(ExtractFeature(model_w, TVT))
      print('\n=========> Test on dataset: {} <=========\n'.format(name))
      test_set.eval(
        normalize_feat=cfg.normalize_feature,
        verbose=True)

  def validate():
    if val_set.extract_feat_func is None:
      val_set.set_feat_func(ExtractFeature(model_w, TVT))
    print('\n=========> Test on validation set <=========\n')
    mAP, cmc_scores, _, _ = val_set.eval(
      normalize_feat=cfg.normalize_feature,
      to_re_rank=False,
      verbose=False)
    print()
    return mAP, cmc_scores[0]

  if cfg.only_test:
    test(load_model_weight=True)
    return

  ############
  # Training #
  ############

  scheduler_Di,scheduler_Dp,scheduler_Dt,scheduler_G = scheduler(optimizer_Di,optimizer_Dp,optimizer_Dt,optimizer_G)

  start_ep = resume_ep if cfg.resume else 0


  for ep in range(start_ep, cfg.total_epochs):

    # Adjust Learning Rate
    if cfg.lr_decay_type == 'exp':
      adjust_lr_exp(
        optimizer,
        cfg.base_lr,
        ep + 1,
        cfg.total_epochs,
        cfg.exp_decay_at_epoch)
    else:
      adjust_lr_staircase(
        optimizer,
        cfg.base_lr,
        ep + 1,
        cfg.staircase_decay_at_epochs,
        cfg.staircase_decay_multiply_factor)

    scheduler_Di.step()
    scheduler_Dp.step()
    scheduler_Dt.step()
    scheduler_G.step()


    may_set_mode(all_model_optims, 'train')
    model_net_E.apply(set_bn_fix)
    model_net_Di.apply(set_bn_fix)

    # For recording precision, satisfying margin, etc
    prec_meter = AverageMeter()
    sm_meter = AverageMeter()
    dist_ap_meter = AverageMeter()
    dist_an_meter = AverageMeter()
    loss_meter = AverageMeter()

    ep_st = time.time()
    step = 0
    epoch_done = False
    while not epoch_done:

      step += 1
      step_st = time.time()

      ims, im_names, labels, pose,pose_target,mirrored, epoch_done = train_set.next_batch()
      noise = torch.randn(cfg.ids_per_batch*cfg.ims_per_id, cfg.noise_feature_size)
      


      ims_var = Variable(TVT(torch.from_numpy(ims).float()))
      pose_var = Variable(TVT(torch.from_numpy(pose).float()))
      pose_target_var = Variable(TVT(torch.from_numpy(pose_target).float()))
      labels_t = TVT(torch.from_numpy(labels).long())
      
      noise_var = Variable(TVT(noise.float()))
      
      '--C--get truth-picture-feature'
      feat = model_w(ims_var)
      #print(pose_var.device, feat.device, noise_var.device)
      #print(feat.size) 
      
      #visualizer.display_current_results(get_current_visuals(ims_var,pose_target_var,trans_picture.detach(),pose_var), ep, 1)

      'the C triplet loss with truth picture trule label'
      loss1, p_inds, n_inds, dist_ap, dist_an, dist_mat = global_loss(
        tri_loss, feat, labels_t,
        normalize_feature=cfg.normalize_feature)


      p_picture,p_nosie,p_pose,n_picture,n_nosie,n_pose = get_the_pn_truth_pose_nosie(p_inds,n_inds,ims,noise,pose)


#'-----------------------------------update FD-GAN----------------------------' 
      labels_v_var,origin_var,bs,B_map_var,GAN_target_var,z = get_the_FDGAN_input(cfg,labels,ims,pose,pose_target,noise)      
      #pdb.set_trace()
      '--FD--GAN--forward--'
      A_id1,A_id2,id_score = model_net_E_w(origin_var[:bs//2],origin_var[bs//2:])
      A_id = torch.cat((A_id1, A_id2))
      GAN_trans_picture = model_net_G_w(B_map_var, A_id.view(A_id.size(0), A_id.size(1), 1, 1), z.view(z.size(0), z.size(1), 1, 1))

      '--------------------------------------------------------------------------------'



      'backward_DP'
      real_pose = torch.cat((B_map_var, GAN_target_var),dim=1)
      fake_pose = torch.cat((B_map_var,GAN_trans_picture.detach()),dim=1)
      pred_real = model_net_Dp_w(real_pose)
      pred_fake = model_net_Dp_w(fake_pose)

      if random.choice(rand_list):
          loss_D_real = GAN_D_loss(pred_fake, True)
          loss_D_fake = GAN_D_loss(pred_real, False)
      else:
          loss_D_real = GAN_D_loss(pred_real, True)
          loss_D_fake = GAN_D_loss(pred_fake, False)
      loss_D = (loss_D_real + loss_D_fake) * 0.5

      optimizer_Dp.zero_grad()
      loss_D.backward()
      optimizer_Dp.step()


      'backward_Di'
      _,_,pred_real_Di = model_net_Di_w(origin_var,GAN_target_var)
      _,_,pred_fake_Di = model_net_Di_w(origin_var,GAN_trans_picture.detach())
      if random.choice(rand_list):
          loss_D_Di_real = GAN_D_loss(pred_fake_Di, True)
          loss_D_Di_fake = GAN_D_loss(pred_real_Di, False)
      else:
          loss_D_Di_real = GAN_D_loss(pred_real_Di, True)
          loss_D_Di_fake = GAN_D_loss(pred_fake_Di, False)
      loss_D_Di = (loss_D_Di_real + loss_D_Di_fake) * 0.5
      optimizer_Di.zero_grad()
      loss_D_Di.backward()
      optimizer_Di.step()     



      
    #'--------------updata triplet gan D------------------'
    #'--------------------forward------------------------'
      p_picture_var = Variable(TVT(torch.from_numpy(p_picture).float()))
      p_nosie_var = Variable(TVT(torch.from_numpy(p_nosie).float()))
      p_pose_var = Variable(TVT(torch.from_numpy(p_pose).float()))
      ps = p_picture_var.size(0)

      n_picture_var = Variable(TVT(torch.from_numpy(n_picture).float()))
      n_nosie_var = Variable(TVT(torch.from_numpy(n_nosie).float()))
      n_pose_var = Variable(TVT(torch.from_numpy(n_pose).float()))
      ns = n_picture_var.size(0)
            

            
      p_id1,p_id2,_ = model_net_E_w(p_picture_var[:ps//2],p_picture_var[ps//2:])
      p_id = torch.cat((p_id1, p_id2))
      p_trans_picture = model_net_G_w(p_pose_var, p_id.view(p_id.size(0), p_id.size(1), 1, 1), p_nosie_var.view(p_nosie_var.size(0), p_nosie_var.size(1), 1, 1))

      n_id1,n_id2,_ = model_net_E_w(n_picture_var[:ns//2],n_picture_var[ns//2:])
      n_id = torch.cat((n_id1, n_id2))
      n_trans_picture = model_net_G_w(n_pose_var, n_id.view(n_id.size(0), n_id.size(1), 1, 1), n_nosie_var.view(n_nosie_var.size(0), n_nosie_var.size(1), 1, 1))






      an_po_label_grad = dist_ap-dist_ap.detach()+Variable(TVT(torch.ones_like(torch.empty(dist_ap.shape[0]))))
      an_po_label_grad_suit = Variable(TVT(torch.ones_like(torch.empty(1,1,ims.shape[2],ims.shape[3])))).fill_(an_po_label_grad[0])
      for i in range (dist_ap.shape[0]-1):
        an_po_label_grad_suit = torch.cat((an_po_label_grad_suit,Variable(TVT(torch.ones_like(torch.empty(1,1,ims.shape[2],ims.shape[3])))).fill_(an_po_label_grad[i+1])),dim=0)


      an_ng_label_grad = dist_an-dist_an.detach()
      an_ng_label_grad_suit = Variable(TVT(torch.ones_like(torch.empty(1,1,ims.shape[2],ims.shape[3])))).fill_(an_ng_label_grad[0])
      for i in range (dist_an.shape[0]-1):
        an_ng_label_grad_suit = torch.cat((an_ng_label_grad_suit,Variable(TVT(torch.ones_like(torch.empty(1,1,ims.shape[2],ims.shape[3])))).fill_(an_ng_label_grad[i+1])),dim=0)





      a_p = torch.cat((an_po_label_grad_suit,ims_var),dim=1)
      a_n = torch.cat((an_ng_label_grad_suit,ims_var),dim=1)
      po_truth = torch.cat((a_p.detach(),Variable(TVT(torch.from_numpy(p_picture).float()))),dim=1)
      po_fake = torch.cat((a_p.detach(),p_trans_picture.detach()),dim=1)
      ne_truth = torch.cat((a_n.detach(),Variable(TVT(torch.from_numpy(n_picture).float()))),dim=1)
      ne_fake = torch.cat((a_n.detach(),n_trans_picture.detach()),dim=1)

      pred_a_p_t = model_net_Dt_w(po_truth)
      pred_a_p_f = model_net_Dt_w(po_fake)
      pred_a_n_t = model_net_Dt_w(ne_truth)
      pred_a_n_f = model_net_Dt_w(ne_fake)
      if random.choice(rand_list):
        loss_D_Dt_p_real = GAN_D_loss(pred_a_p_f, True)
        loss_D_Dt_p_fake = GAN_D_loss(pred_a_p_t, False)
        loss_D_Dt_n_real = GAN_D_loss(pred_a_n_f, True)
        loss_D_Dt_n_fake = GAN_D_loss(pred_a_n_t, False)
      else:
        loss_D_Dt_p_real = GAN_D_loss(pred_a_p_t, True)
        loss_D_Dt_p_fake = GAN_D_loss(pred_a_p_f, False)
        loss_D_Dt_n_real = GAN_D_loss(pred_a_n_t, True)
        loss_D_Dt_n_fake = GAN_D_loss(pred_a_n_f, False)
      loss_D_Dt = (loss_D_Dt_p_real + loss_D_Dt_p_fake+loss_D_Dt_n_real+loss_D_Dt_n_fake) * 0.25
      optimizer_Dt.zero_grad()
      loss_D_Dt.backward()
      optimizer_Dt.step()




      'backward_G/E'
      '----the loss from the FD-GAN D'
      loss_v = F.cross_entropy(id_score,labels_v_var.view(-1))
      loss_r = F.l1_loss(GAN_trans_picture,GAN_target_var)
      GAN_trans_picture_1 = GAN_trans_picture[:GAN_trans_picture.size(0)//2]
      GAN_trans_picture_2 = GAN_trans_picture[GAN_trans_picture.size(0)//2:]
      loss_sp = F.l1_loss(
        GAN_trans_picture_1[labels_v_var.view(labels_v_var.size(0),1,1,1).expand_as(GAN_trans_picture_1)==1],
        GAN_trans_picture_2[labels_v_var.view(labels_v_var.size(0),1,1,1).expand_as(GAN_trans_picture_1)==1]

      )
      _,_,pred_fake_Di_G= model_net_Di_w(origin_var,GAN_trans_picture)
      pred_fake_G = model_net_Dp_w(torch.cat((B_map_var,GAN_trans_picture),dim=1))

      loss_G_GAN_Di = GAN_G_loss(pred_fake_Di_G,True)
      loss_G_GAN_Dp = GAN_G_loss(pred_fake_G,True)
      '---the loss from the triplet-GAN D'
      pred_a_p_f_G = model_net_Dt_w(torch.cat((a_p.detach(),p_trans_picture),dim=1))
      pred_a_n_f_G = model_net_Dt_w(torch.cat((a_n.detach(),n_trans_picture),dim=1))



      loss_G_GAN_D_Dt_p_fake = GAN_G_loss(pred_a_p_f_G, True)
      loss_G_GAN_D_Dt_n_fake = GAN_G_loss(pred_a_n_f_G, True)
      #loss_GE = loss_G_GAN_Di+loss_G_GAN_Dp+loss_v+loss_r+loss_sp
      loss_GE = loss_G_GAN_D_Dt_n_fake+loss_G_GAN_D_Dt_p_fake+loss_G_GAN_Di+loss_G_GAN_Dp+loss_v+loss_r+loss_sp
      optimizer_G.zero_grad()
      loss_GE.backward()
      optimizer_G.step()
      
    


      '-----------------------------update triplet-GAN C-------------'
      'the C triplet loss with fake picture trule label'
      '--FDGAN the E get feature--'
      trans_feat1,trans_feat2,_ = model_net_E_w(ims_var[:ims_var.size(0)//2],ims_var[ims_var.size(0)//2:])# the feature used by trans
      trans_feat = torch.cat((trans_feat1,trans_feat2))
      
      '--FDGAN the G get trans picture'
      trans_picture = model_net_G_w(pose_var,trans_feat.view(trans_feat.size(0),trans_feat.size(1),1,1),noise_var.view(noise_var.size(0),noise_var.size(1),1,1))#  False a
      '--C--get trans-picture-feature'
      feat2 = model_w(trans_picture)
      loss2, p_inds2, n_inds2, dist_ap2, dist_an2, dist_mat2 = global_loss(
        tri_loss, feat2, labels_t,
        normalize_feature=cfg.normalize_feature)
      '--the loss from triplet-GAN D'

      


      pred_a_p_t_C = model_net_Dt_w(torch.cat((a_p,Variable(TVT(torch.from_numpy(p_picture).float()))),dim=1))
      pred_a_n_t_C = model_net_Dt_w(torch.cat((a_n,Variable(TVT(torch.from_numpy(n_picture).float()))),dim=1))


    
      
      loss_C_ap = GAN_G_loss(pred_a_p_t_C,False)
      loss_C_an = GAN_G_loss(pred_a_n_t_C,False)
      loss = loss1+loss2+loss_C_ap+loss_C_an
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()


      ############
      # Step Log #
      ############

      # precision
      prec = (dist_an > dist_ap).data.float().mean()
      # the proportion of triplets that satisfy margin
      sm = (dist_an > dist_ap + cfg.margin).data.float().mean()
      # average (anchor, positive) distance
      d_ap = dist_ap.data.mean()
      # average (anchor, negative) distance
      d_an = dist_an.data.mean()

      prec_meter.update(prec)
      sm_meter.update(sm)
      dist_ap_meter.update(d_ap)
      dist_an_meter.update(d_an)
      loss_meter.update(to_scalar(loss))

      if step % cfg.steps_per_log == 0:
        time_log = '\tStep {}/Ep {}, {:.2f}s'.format(
          step, ep + 1, time.time() - step_st, )

        tri_log = (', prec {:.2%}, sm {:.2%}, '
                   'd_ap {:.4f}, d_an {:.4f}, '
                   'loss {:.4f}'.format(
          prec_meter.val, sm_meter.val,
          dist_ap_meter.val, dist_an_meter.val,
          loss_meter.val, ))

        log = time_log + tri_log
        print(log)


    #############
    # Epoch Log #
    #############

    time_log = 'Ep {}, {:.2f}s'.format(ep + 1, time.time() - ep_st)

    tri_log = (', prec {:.2%}, sm {:.2%}, '
               'd_ap {:.4f}, d_an {:.4f}, '
               'loss {:.4f}'.format(
      prec_meter.avg, sm_meter.avg,
      dist_ap_meter.avg, dist_an_meter.avg,
      loss_meter.avg, ))

    log = time_log + tri_log
    print(log)

    ##########################
    # Test on Validation Set #
    ##########################

    mAP, Rank1 = 0, 0
    if ((ep + 1) % cfg.epochs_per_val == 0) and (val_set is not None):
      mAP, Rank1 = validate()

    # Log to TensorBoard

    if cfg.log_to_file:
      if writer is None:
        writer = SummaryWriter(log_dir=osp.join(cfg.exp_dir, 'tensorboard'))
      writer.add_scalars(
        'val scores',
        dict(mAP=mAP,
             Rank1=Rank1),
        ep)
      writer.add_scalars(
        'loss',
        dict(loss=loss_meter.avg, ),
        ep)
      writer.add_scalars(
        'precision',
        dict(precision=prec_meter.avg, ),
        ep)
      writer.add_scalars(
        'satisfy_margin',
        dict(satisfy_margin=sm_meter.avg, ),
        ep)
      writer.add_scalars(
        'average_distance',
        dict(dist_ap=dist_ap_meter.avg,
             dist_an=dist_an_meter.avg, ),
        ep)

    # save ckpt
    if cfg.log_to_file:
      save_ckpt(modules_optims, ep+1, 0, cfg.ckpt_file)
      save_network(cfg.user_dir+'new_reid/Result/tlg',model_net_Di_w,'Di',ep+1)
      save_network(cfg.user_dir+'new_reid/Result/tlg',model_net_Dp_w,'Dp',ep+1)
      save_network(cfg.user_dir+'new_reid/Result/tlg',model_net_Dt_w,'Dt',ep+1)
      save_network(cfg.user_dir+'new_reid/Result/tlg',model_net_G_w,'G',ep+1)
      save_network(cfg.user_dir+'new_reid/Result/tlg',model_net_E_w,'E',ep+1)
      save_ckpt(all_optims, ep+1, 0, cfg.user_dir+'new_reid/Result/tlg/optim.pth')


  ########
  # Test #
  ########

  test(load_model_weight=False)


if __name__ == '__main__':
  main()
