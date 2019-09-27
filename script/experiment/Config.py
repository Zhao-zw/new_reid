import argparse
import time
import numpy as np
import os.path as osp
import torch
from tri_loss.utils.utils import str2bool
from tri_loss.utils.utils import time_str
from torch.autograd import Variable




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
    parser.add_argument('--ids_per_batch', type=int, default=6)   # 32
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
    parser.add_argument('--user_dir', type=str, default='/home/zzw/Desktop/')
    parser.add_argument('--model_weight_file', type=str, default='')

    parser.add_argument('--base_lr', type=float, default=2e-4)
    parser.add_argument('--lr_decay_type', type=str, default='exp',
                        choices=['exp', 'staircase'])
    parser.add_argument('--exp_decay_at_epoch', type=int, default=151)
    parser.add_argument('--staircase_decay_at_epochs',
                        type=eval, default=(101, 201,))
    parser.add_argument('--staircase_decay_multiply_factor',
                        type=float, default=0.1)
    parser.add_argument('--total_epochs', type=int, default=500)
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
    parser.add_argument('--use_FDGAN_module', type=str2bool, default=True)
    parser.add_argument('--display_single_pane_ncols', type=int, default=0, help='if positive, display all images in a single visdom web panel with certain number of images per row.')

    args = parser.parse_args()
    self.user_dir =args.user_dir
    self.use_FDGAN_model = args.use_FDGAN_module
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