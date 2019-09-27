from __future__ import print_function

import sys

sys.path.insert(0, '.')
import numpy as np
import torch
from torch.autograd import Variable
import torch.optim as optim
from torch.nn.parallel import DataParallel

import time
import os.path as osp
from tensorboardX import SummaryWriter
import numpy as np
import argparse


from FDGAN.utils.visualizer import Visualizer
from FDGAN.model import create
from FDGAN.model.embedding import EltwiseSubEmbed
from FDGAN.model.multi_branch import SiameseNet


from tri_loss.dataset import create_dataset
from tri_loss.model.Model import Model
from tri_loss.model.TripletLoss import TripletLoss
from tri_loss.model.loss import global_loss
from tri_loss.model.PoseGenerator import get_norm_layer, CustomPoseGenerator, init_weights,NLayerDiscriminator,_load_state_dict,get_current_visuals
from torch.nn import functional as F

from tri_loss.utils.utils import time_str
from tri_loss.utils.utils import str2bool
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
import pdb


class Config(object):
  def __init__(self):

    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='zzw_NB', help='directory to save models')
    parser.add_argument('-d', '--sys_device_ids', type=eval, default=(0,))
    parser.add_argument('-r', '--run', type=int, default=1)
    parser.add_argument('--set_seed', type=str2bool, default=False)
    parser.add_argument('--dataset', type=str, default='market1501',
                        choices=['market1501', 'cuhk03', 'duke', 'combined'])
    parser.add_argument('--trainset_part', type=str, default='trainval',
                        choices=['trainval', 'train'])

    parser.add_argument('--resize_h_w', type=eval, default=(256, 128))
    # These several only for training set
    parser.add_argument('--crop_prob', type=float, default=0)
    parser.add_argument('--crop_ratio', type=float, default=1)
    parser.add_argument('--mirror', type=str2bool, default=True)
    parser.add_argument('--ids_per_batch', type=int, default=16)   # 32
    parser.add_argument('--ims_per_id', type=int, default=4)

    parser.add_argument('--log_to_file', type=str2bool, default=True)
    parser.add_argument('--steps_per_log', type=int, default=1)
    parser.add_argument('--epochs_per_val', type=int, default=1e10)

    parser.add_argument('--last_conv_stride', type=int, default=1,
                        choices=[1, 2])
    parser.add_argument('--normalize_feature', type=str2bool, default=False)
    parser.add_argument('--margin', type=float, default=0.3)

    parser.add_argument('--only_test', type=str2bool, default=False)
    parser.add_argument('--resume', type=str2bool, default=True)
    parser.add_argument('--exp_dir', type=str, default='')
    parser.add_argument('--model_weight_file', type=str, default='')

    parser.add_argument('--base_lr', type=float, default=2e-4)
    parser.add_argument('--lr_decay_type', type=str, default='exp',
                        choices=['exp', 'staircase'])
    parser.add_argument('--exp_decay_at_epoch', type=int, default=151)
    parser.add_argument('--staircase_decay_at_epochs',
                        type=eval, default=(101, 201,))
    parser.add_argument('--staircase_decay_multiply_factor',
                        type=float, default=0.1)
    parser.add_argument('--total_epochs', type=int, default=200)
    parser.add_argument('--pose_feature_size', type=int, default=128, help='length of feature vector for pose')
    parser.add_argument('--noise_feature_size', type=int, default=256, help='length of feature vector for noise')
    parser.add_argument('--drop', type=float, default=0.2, help='dropout for the netG')
    parser.add_argument('--norm', type=str, default='batch', help='instance normalization or batch normalization')
    parser.add_argument('--fuse_mode', type=str, default='cat', help='method to fuse reid feature and pose feature [cat|add]')
    parser.add_argument('--connect_layers', type=int, default=0, help='skip connections num for netG')
    parser.add_argument('--display_id', type=int, default=1, help='window id of the web display, set 0 for non-usage of visdom')
    parser.add_argument('--no_html', action='store_true', help='do not save intermediate training results to [opt.checkpoints]/name/web/')
    parser.add_argument('--display_winsize', type=int, default=256,  help='display window size')
    parser.add_argument('--display_port', type=int, default=6006, help='visdom port of the web display')
    parser.add_argument('--checkpoints', type=str, default='./htm/', help='root pa	th to save htmls')
    parser.add_argument('--display_single_pane_ncols', type=int, default=0, help='if positive, display all images in a single visdom web panel with certain number of images per row.')

    args = parser.parse_args()
    self.display_single_pane_ncols = args.display_single_pane_ncols
    self.checkpoints = args.checkpoints
    self.display_port = args.display_port
    self.no_html = args.no_html
    self.display_winsize =args.display_winsize
    self.name = args.name
    # gpu ids
    self.sys_device_ids = args.sys_device_ids

    # If you want to make your results exactly reproducible, you have
    # to fix a random seed.
    if args.set_seed:
      self.seed = 1
    else:
      self.seed = None

    # The experiments can be run for several times and performances be averaged.
    # `run` starts from `1`, not `0`.
    self.run = args.run

    ###########
    # Dataset #
    ###########

    # If you want to make your results exactly reproducible, you have
    # to also set num of threads to 1 during training.
    if self.seed is not None:
      self.prefetch_threads = 1
    else:
      self.prefetch_threads = 2

    self.dataset = args.dataset
    self.trainset_part = args.trainset_part

    # Image Processing

    # Just for training set
    self.crop_prob = args.crop_prob
    self.crop_ratio = args.crop_ratio
    self.resize_h_w = args.resize_h_w

    # Whether to scale by 1/255
    self.scale_im = True
    self.im_mean = [0.486, 0.459, 0.408]
    self.im_std = [0.229, 0.224, 0.225]

    self.train_mirror_type = 'random' if args.mirror else None

    self.ids_per_batch = args.ids_per_batch
    self.ims_per_id = args.ims_per_id
    self.train_final_batch = False
    self.train_shuffle = True

    self.test_batch_size = 32
    self.test_final_batch = True
    self.test_mirror_type = None
    self.test_shuffle = False

    dataset_kwargs = dict(
      name=self.dataset,
      resize_h_w=self.resize_h_w,
      scale=self.scale_im,
      im_mean=self.im_mean,
      im_std=self.im_std,
      batch_dims='NCHW',
      num_prefetch_threads=self.prefetch_threads)

    prng = np.random
    if self.seed is not None:
      prng = np.random.RandomState(self.seed)
    self.train_set_kwargs = dict(
      part=self.trainset_part,
      ids_per_batch=self.ids_per_batch,
      ims_per_id=self.ims_per_id,
      final_batch=self.train_final_batch,
      shuffle=self.train_shuffle,
      crop_prob=self.crop_prob,
      crop_ratio=self.crop_ratio,
      mirror_type=self.train_mirror_type,
      prng=prng)
    self.train_set_kwargs.update(dataset_kwargs)

    prng = np.random
    if self.seed is not None:
      prng = np.random.RandomState(self.seed)
    self.val_set_kwargs = dict(
      part='val',
      batch_size=self.test_batch_size,
      final_batch=self.test_final_batch,
      shuffle=self.test_shuffle,
      mirror_type=self.test_mirror_type,
      prng=prng)
    self.val_set_kwargs.update(dataset_kwargs)

    prng = np.random
    if self.seed is not None:
      prng = np.random.RandomState(self.seed)
    self.test_set_kwargs = dict(
      part='test',
      batch_size=self.test_batch_size,
      final_batch=self.test_final_batch,
      shuffle=self.test_shuffle,
      mirror_type=self.test_mirror_type,
      prng=prng)
    self.test_set_kwargs.update(dataset_kwargs)

    ###############
    # ReID Model  #
    ###############

    # The last block of ResNet has stride 2. We can set the stride to 1 so that
    # the spatial resolution before global pooling is doubled.
    self.last_conv_stride = args.last_conv_stride

    # Whether to normalize feature to unit length along the Channel dimension,
    # before computing distance
    self.normalize_feature = args.normalize_feature

    # Margin of triplet loss
    self.margin = args.margin

    #############
    # Training  #
    #############

    self.weight_decay = 0.0005

    # Initial learning rate
    self.base_lr = args.base_lr
    self.lr_decay_type = args.lr_decay_type
    self.exp_decay_at_epoch = args.exp_decay_at_epoch
    self.staircase_decay_at_epochs = args.staircase_decay_at_epochs
    self.staircase_decay_multiply_factor = args.staircase_decay_multiply_factor
    # Number of epochs to train
    self.total_epochs = args.total_epochs

    # How often (in epochs) to test on val set.
    self.epochs_per_val = args.epochs_per_val

    # How often (in batches) to log. If only need to log the average
    # information for each epoch, set this to a large value, e.g. 1e10.
    self.steps_per_log = args.steps_per_log

    # Only test and without training.
    self.only_test = args.only_test

    self.resume = args.resume

    #######
    # Log #
    #######

    # If True,
    # 1) stdout and stderr will be redirected to file,
    # 2) training loss etc will be written to tensorboard,
    # 3) checkpoint will be saved
    self.log_to_file = args.log_to_file

    # The root dir of logs.
    if args.exp_dir == '':
      self.exp_dir = osp.join(
        'exp/train',
        '{}'.format(self.dataset),
        #
        'lcs_{}_'.format(self.last_conv_stride) +
        ('nf_' if self.normalize_feature else 'not_nf_') +
        'margin_{}_'.format(tfs(self.margin)) +
        'lr_{}_'.format(tfs(self.base_lr)) +
        '{}_'.format(self.lr_decay_type) +
        ('decay_at_{}_'.format(self.exp_decay_at_epoch)
         if self.lr_decay_type == 'exp'
         else 'decay_at_{}_factor_{}_'.format(
          '_'.join([str(e) for e in args.staircase_decay_at_epochs]),
          tfs(self.staircase_decay_multiply_factor))) +
        'total_{}'.format(self.total_epochs),
        #
        'run{}'.format(self.run),
      )
    else:
      self.exp_dir = args.exp_dir

    self.stdout_file = osp.join(
      self.exp_dir, 'stdout_{}.txt'.format(time_str()))
    self.stderr_file = osp.join(
      self.exp_dir, 'stderr_{}.txt'.format(time_str()))

    # Saving model weights and optimizer states, for resuming.
    self.ckpt_file = osp.join(self.exp_dir, 'ckpt.pth')
    # Just for loading a pretrained model; no optimizer states is needed.
    self.model_weight_file = args.model_weight_file
    self.pose_feature_size = args.pose_feature_size
    self.noise_feature_size = args.noise_feature_size
    self.drop = args.drop
    self.norm = args.norm
    self.fuse_mode = args.fuse_mode
    self.connect_layers = args.connect_layers
    self.display_id = args.display_id


class ExtractFeature(object):
  """A function to be called in the val/test set, to extract features.
  Args:
    TVT: A callable to transfer images to specific device.
  """

  def __init__(self, model, TVT):
    self.model = model
    self.TVT = TVT

  def __call__(self, ims):
    old_train_eval_model = self.model.training
    # Set eval mode.
    # Force all BN layers to use global mean and variance, also disable
    # dropout.
    self.model.eval()
    ims = Variable(self.TVT(torch.from_numpy(ims).float()))
    feat = self.model(ims)
    feat = feat.data.cpu().numpy()
    # Restore the model to its old train/eval mode.
    self.model.train(old_train_eval_model)
    return feat


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
    train_set = create_dataset(**cfg.train_set_kwargs)
    # The combined dataset does not provide val set currently.
    val_set = None if cfg.dataset == 'combined' else create_dataset(**cfg.val_set_kwargs)

  test_sets = []
  test_set_names = []
  if cfg.dataset == 'combined':
    for name in ['market1501', 'cuhk03', 'duke']:
      cfg.test_set_kwargs['name'] = name
      test_sets.append(create_dataset(**cfg.test_set_kwargs))
      test_set_names.append(name)
  else:
    test_sets.append(create_dataset(**cfg.test_set_kwargs))
    test_set_names.append(cfg.dataset)

  ###########
  # Models  #
  ###########

  visualizer = Visualizer(cfg)

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

  _load_state_dict(model_net_G,'/home/zzw/Desktop/reid_cvpr/model_net/best_net_G.pth')
  _load_state_dict(model_net_E,'/home/zzw/Desktop/reid_cvpr/model_net/best_net_E.pth')
  #pdb.set_trace()
  #init_weights(model_net_Di)
  #init_weights(model_net_Dp)
  # Model wrapper
  model_w = DataParallel(model).cuda()

  model_net_G_w =  DataParallel(model_net_G).cuda()
  model_net_E_w = DataParallel(model_net_E).cuda()
  #model_net_Di_w =  DataParallel(model_net_Di).cuda()
  #model_net_Dp_w =  DataParallel(model_net_Dp).cuda()


  #############################
  # Criteria and Optimizers   #
  #############################

  tri_loss = TripletLoss(margin=cfg.margin)

  optimizer = optim.Adam(model.parameters(),
                         lr=cfg.base_lr,
                         weight_decay=cfg.weight_decay)

  # Bind them together just to save some codes in the following usage.
  modules_optims = [model, optimizer]

  ################################
  # May Resume Models and Optims #
  ################################

  if cfg.resume:
    resume_ep, scores = load_ckpt(modules_optims, cfg.ckpt_file)

  # May Transfer Models and Optims to Specified Device. Transferring optimizer
  # is to cope with the case when you load the checkpoint to a new device.
  TMO(modules_optims)

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

    may_set_mode(modules_optims, 'train')

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
      noise = torch.randn(64, cfg.noise_feature_size)
      


      ims_var = Variable(TVT(torch.from_numpy(ims).float()))
      pose_var = Variable(TVT(torch.from_numpy(pose).float()))
      pose_target_var = Variable(TVT(torch.from_numpy(pose_target).float()))
      labels_t = TVT(torch.from_numpy(labels).long())
      
      noise_var = Variable(TVT(noise.float()))
      

      feat = model_w(ims_var)
      #print(pose_var.device, feat.device, noise_var.device)
      trans_feat1,trans_feat2,_ = model_net_E_w(ims_var[:ims_var.size(0)//2],ims_var[ims_var.size(0)//2:])# the feature used by trans
      trans_feat = torch.cat((trans_feat1,trans_feat2))
      
      trans_picture = model_net_G_w(pose_var,trans_feat.view(trans_feat.size(0),trans_feat.size(1),1,1),noise_var.view(noise_var.size(0),noise_var.size(1),1,1))#  False a
      #print(feat.size) 
      
      #visualizer.display_current_results(get_current_visuals(ims_var,pose_target_var,trans_picture.detach(),pose_var), ep, 1)
      feat2 = model_w(trans_picture)

      loss1, p_inds, n_inds, dist_ap, dist_an, dist_mat = global_loss(
        tri_loss, feat, labels_t,
        normalize_feature=cfg.normalize_feature)

      loss2, p_inds1, n_inds1, dist_ap1, dist_an1, dist_mat1 = global_loss(
        tri_loss, feat2, labels_t,
        normalize_feature=cfg.normalize_feature)
      
      loss = loss1*0.8+loss2*0.2
      '''
      p_picture = []
      for p_ind in p_inds:
        p_picture.append(ims[p_ind])
      p_picture = np.array(p_picture)

      p_trans_picture = []
      for p_ind in p_inds:
        p_trans_picture.append(trans_picture[p_ind].data.cpu().numpy())
      p_trans_picture = np.array(p_trans_picture) 

      n_picture = []
      for n_ind in n_inds:
        n_picture.append(ims[n_ind])
      n_picture = np.array(n_picture)

      n_trans_picture = []
      for n_ind in n_inds:
        n_trans_picture.append(trans_picture[n_ind].data.cpu().numpy())
      n_trans_picture = np.array(n_trans_picture) 

#'-----------------------------------update GAN' 
      data_random_state = np.random.get_state()
      input1_label = labels.copy()
      input2_label = labels.copy()
      np.random.shuffle(input2_label)
      labels_v = (torch.from_numpy(input1_label).long() == torch.from_numpy(input2_label).long()).long()
      labels_v_var = TVT(labels_v.long())
      
      
      input1_ims = ims.copy()   
      input2_ims = ims.copy()
      np.random.set_state(data_random_state)
      np.random.shuffle(input2_ims)
      origin = torch.cat([torch.from_numpy(input1_ims).float(),torch.from_numpy(input2_ims).float()])
      origin_var = Variable(TVT(origin))
      bs = origin_var.size(0)

      input1_map = pose.copy()
      input2_map = pose.copy()  
      np.random.set_state(data_random_state)
      np.random.shuffle(input2_map)
      mask = labels_v.view(-1,1,1,1).expand_as(torch.from_numpy(input1_map))
      input2_map = torch.from_numpy(input1_map).float()*mask.float() + torch.from_numpy(input2_map).float()*(1-mask.float())
      B_map = torch.cat([torch.from_numpy(input1_map).float(),input2_map])
      B_map_var = Variable(TVT(B_map.float()))
      pdb.set_trace()
    #,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,
      targetss = torch.from_numpy(pose_target)
      target = torch.cat([targetss,targetss])
      target_var = Variable(TVT(target.float()))

      noisess = torch.cat((noise,noise))
      z = Variable(TVT(noisess.float()))
 

      
      #pdb.set_trace()

      A_id1,A_id2,id_score = model_net_E_w(origin_var[:bs//2],origin_var[bs//2:])
      A_id = torch.cat((A_id1, A_id2))
      GAN_trans_picture = model_net_G_w(B_map_var, A_id.view(A_id.size(0), A_id.size(1), 1, 1), z.view(z.size(0), z.size(1), 1, 1))


    #"-------------------"
      real_pose = torch.cat((B_map_var, target_var),dim=1)
      pred_real = model_net_Dp_w(real_pose)
      '''
      
      
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
      save_ckpt(modules_optims, ep + 1, 0, cfg.ckpt_file)

  ########
  # Test #
  ########

  test(load_model_weight=False)


if __name__ == '__main__':
  main()