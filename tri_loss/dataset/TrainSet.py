from .Dataset import Dataset
from ..utils.dataset_utils import parse_im_name
import torch
import os.path as osp
from script.experiment.Config import Config

ospj = osp.join
ospeu = osp.expanduser
from PIL import Image
import numpy as np
from collections import defaultdict
import random
from scipy import ndimage
import pdb


class TrainSet(Dataset):
  """Training set for triplet loss.
  Args:
    ids2labels: a dict mapping ids to labels
  """

  def __init__(
      self,
      im_dir=None,
      im_names=None,
      ids2labels=None,
      ids_per_batch=None,
      ims_per_id=None,
      **kwargs):

    # The im dir of all images
    self.im_dir = im_dir
    self.im_names = im_names
    self.ids2labels = ids2labels
    self.ids_per_batch = ids_per_batch
    self.ims_per_id = ims_per_id
    self.pose_aug = 'no'

    im_ids = [parse_im_name(name, 'id') for name in im_names]
    #pdb.set_trace()
    self.ids_to_im_inds = defaultdict(list)
    for ind, id in enumerate(im_ids):
      self.ids_to_im_inds[id].append(ind)
    #pdb.set_trace()
    self.ids = self.ids_to_im_inds.keys()

    super(TrainSet, self).__init__(
      dataset_size=len(self.ids),
      batch_size=ids_per_batch,
      **kwargs)

  def get_sample(self, ptr):
    """Here one sample means several images (and labels etc) of one id.
    Returns:
      ims: a list of images
    """
    #pdb.set_trace()
    inds = self.ids_to_im_inds[self.ids[ptr]]
    if len(inds) < self.ims_per_id:
      inds = np.random.choice(inds, self.ims_per_id, replace=True)
    else:
      inds = np.random.choice(inds, self.ims_per_id, replace=False)
    im_names = [self.im_names[ind] for ind in inds]
    ims = [np.asarray(Image.open(osp.join(self.im_dir, name)))
           for name in im_names]
    ims, mirrored = zip(*[self.pre_process_im(im) for im in ims])
    labels = [self.ids2labels[self.ids[ptr]] for _ in range(self.ims_per_id)]
    #pdb.set_trace()
    return ims, im_names, labels, mirrored

  def next_batch(self):
    """Next batch of images and labels.
    Returns:
      ims: numpy array with shape [N, H, W, C] or [N, C, H, W], N >= 1
      img_names: a numpy array of image names, len(img_names) >= 1
      labels: a numpy array of image labels, len(labels) >= 1
      mirrored: a numpy array of booleans, whether the images are mirrored
      self.epoch_done: whether the epoch is over
    """
    # Start enqueuing and other preparation at the beginning of an epoch.
    cfg = Config()
    pose_root = osp.join(cfg.user_dir,'new_reid/tri_loss/dataset/Dataset/market1501/poses')
    if self.epoch_done and self.shuffle:
      #np.random.shuffle(self.ids)
      self.ids = list(self.ids)
      np.random.shuffle(self.ids)
    samples, self.epoch_done = self.prefetcher.next_batch()
    im_list, im_names, labels, mirrored = zip(*samples)
    # t = time.time()
    # Transform the list into a numpy array with shape [N, ...]
    ims = np.stack(np.concatenate(im_list))
    # print '---stacking time {:.4f}s'.format(time.time() - t)



    im_names = np.concatenate(im_names)
    labels = np.concatenate(labels)
    mirrored = np.concatenate(mirrored)

    batch_im_ids = [parse_im_name(name, 'id') for name in im_names]
    im_ids = [parse_im_name(name, 'id') for name in self.im_names]
    number_indx = 0
    batch_maps = []
    target_ims = []
    for batch_im_id in batch_im_ids:
      A = [indx for indx, i in enumerate(im_ids) if i == batch_im_id ]
      b = []

      for a in A:
        b.append(self.im_names[a])
      
      if im_names[number_indx] in b :
        b.remove(im_names[number_indx])
        old_pname = random.choice(b)
        t_ims=  np.asarray(Image.open(osp.join(self.im_dir, old_pname)))  


        pname = old_pname[0:9]+old_pname[11:12]+str(int(old_pname[12:13])-1)+'_'+old_pname[18:23]+'txt'
        ppath = osp.join(pose_root,pname)
        landmark = self._load_landmark(ppath, 256/128, 128/64)
        maps = self._generate_pose_map(landmark)
        t_ims,_ = self.pre_process_im(t_ims)
        target_ims.append(t_ims)
        batch_maps.append(maps)
      
      number_indx = number_indx+1
    batch_maps = np.array(batch_maps)
    target_ims = np.array(target_ims)
    #pdb.set_trace()
    return ims, im_names, labels, batch_maps, target_ims,mirrored, self.epoch_done

  def _load_landmark(self, path, scale_h, scale_w):
        landmark = []
        with open(path,'r') as f:
            landmark_file = f.readlines()
        for line in landmark_file:
            line1 = line.strip()
            h0 = int(float(line1.split(' ')[0]) * scale_h)
            w0 = int(float(line1.split(' ')[1]) * scale_w)
            if h0<0: h0=-1
            if w0<0: w0=-1
            landmark.append(torch.Tensor([[h0,w0]]))
        landmark = torch.cat(landmark).long()
        return landmark

  def _generate_pose_map(self, landmark, gauss_sigma=5):
        maps = []
        randnum = landmark.size(0)+1
        if self.pose_aug=='erase':
            randnum = random.randrange(landmark.size(0))
        elif self.pose_aug=='gauss':
            gauss_sigma = random.randint(gauss_sigma-1,gauss_sigma+1)
        elif self.pose_aug!='no':
            assert ('Unknown landmark augmentation method, choose from [no|erase|gauss]')
        for i in range(landmark.size(0)):
            map = np.zeros([256,128])
            if landmark[i,0]!=-1 and landmark[i,1]!=-1 and i!=randnum:
                map[landmark[i,0],landmark[i,1]]=1
                map = ndimage.filters.gaussian_filter(map,sigma = gauss_sigma)
                map = map/map.max()
            maps.append(map)
        maps = np.stack(maps, axis=0)
        return maps