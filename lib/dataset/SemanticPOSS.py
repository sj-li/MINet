import numpy as np
import cv2 as cv
import os

W = 1800 # width of range image
H = 40   # height of range image
LABEL_DICT = {
    0: "unlabeled",
    4: "1 person",
    5: "2+ person",
    6: "rider",
    7: "car",
    8: "trunk",
    9: "plants",
    10: "traffic sign 1", # standing sign
    11: "traffic sign 2", # hanging sign
    12: "traffic sign 3", # high/big hanging sign
    13: "pole",
    14: "trashcan",
    15: "building",
    16: "cone/stone",
    17: "fence",
    21: "bike",
    22: "ground"} # class definition
SEM_COLOR = np.array([
    [0, 0, 0],                       # 0: "unlabeled"
    [0, 0, 0], [0, 0, 0], [0, 0, 0], # don't care
    [30, 30, 255],#[255, 30, 30],                   # 4: "1 person"
    [30, 30, 255], #[255, 30, 30],                   # 5: "2+ person"
    [200, 40, 255],#[255, 40, 200],                  # 6: "rider"
    [245, 150, 100],#[100, 150, 245],                 # 7: "car"
    [0, 60, 135],#[135,60,0],                      # 8: "trunk"
    [0, 175, 0],#[0, 175, 0],                     # 9: "plants"
    [0, 0, 255],#[255, 0, 0],                     # 10: "traffic sign 1"
    [0, 0, 255],#[255, 0, 0],                     # 11: "traffic sign 2"
    [0, 0, 255],#[255, 0, 0],                     # 12: "traffic sign 3"
    [150, 240, 255],#[255, 240, 150],                 # 13: "pole"
    [0, 0, 0],#[125, 255, 0],                   # 14: "trashcan"
    [0, 200, 255],#[255, 200, 0],                   # 15: "building"
    [0, 0, 0],#[50, 255, 255],                  # 16: "cone/stone"
    [50, 120, 255],#[255, 120, 50],                  # 17: "fence"
    [0,0,0],[0,0,0],[0,0,0],         # don't care
    [245, 230, 100],#[100, 230, 245],                 # 21: "bike"
    [75, 0, 175],#[128, 128, 128]
],                # 22: "ground"
    dtype = np.uint8) # color definition

SEM_COLOR_ori = np.array([
    [0, 0, 0],                       # 0: "unlabeled"
    [0, 0, 0], [0, 0, 0], [0, 0, 0], # don't care
    [255, 30, 30],                   # 4: "1 person"
    [255, 30, 30],                   # 5: "2+ person"
    [255, 40, 200],                  # 6: "rider"
    [100, 150, 245],                 # 7: "car"
    [135,60,0],                      # 8: "trunk"
    [0, 175, 0],                     # 9: "plants"
    [255, 0, 0],                     # 10: "traffic sign 1"
    [255, 0, 0],                     # 11: "traffic sign 2"
    [255, 0, 0],                     # 12: "traffic sign 3"
    [255, 240, 150],                 # 13: "pole"
    [125, 255, 0],                   # 14: "trashcan"
    [255, 200, 0],                   # 15: "building"
    [50, 255, 255],                  # 16: "cone/stone"
    [255, 120, 50],                  # 17: "fence"
    [0,0,0],[0,0,0],[0,0,0],         # don't care
    [100, 230, 245],                 # 21: "bike"
    [128, 128, 128]],                # 22: "ground"
    dtype = np.uint8) # color definition

def read_points(bin_file):
    points = np.fromfile(bin_file, dtype = np.float32)
    points = np.reshape(points,(-1,4)) # x,y,z,intensity
    return points

def read_semlabels(label_file):
    semlabels = np.fromfile(label_file, dtype = np.uint32) & 0xffff     ## todo: ? uint32 in KITTI
    return semlabels

def read_inslabels(label_file):
    inslabels = np.fromfile(label_file, dtype = np.uint32) >> 16
    return inslabels

def get_rangeimage(bin_file, tag_file):
    points = read_points(bin_file)                          ## points after tags mask
    tags = np.fromfile(tag_file, dtype = np.bool)           ## all points in a 2D image, e.g. H*W
    dis = np.linalg.norm(points[:,0:3], axis = 1) * 5       ## fixme: why  *5
    dis = np.minimum(dis, 255)
    dis = dis.astype(np.uint8)
    dis_vec = np.zeros((H*W), dtype = np.uint8)
    dis_vec[tags] = dis
    dis_mat = np.reshape(dis_vec, (H,W))
    rangeimage = cv.cvtColor(dis_mat, cv.COLOR_GRAY2BGR)   ## fixme?
    return rangeimage                   # H, W, 3

def get_semimage(label_file, tag_file):
    semlabels = read_semlabels(label_file)
    tags = np.fromfile(tag_file, dtype = np.bool)
    label_vec = np.zeros((H*W), dtype = np.uint32)
    label_vec[tags] = semlabels
    tmp1 = np.reshape(label_vec, (H, W))
    tmp2 = np.resize(tmp1, (40, 1024))
    image_vec = SEM_COLOR[label_vec]
    semimage = np.reshape(image_vec, (H,W,3))
    return semimage

def get_insimage(label_file, tag_file):                 ## todo: no instance label so far
    inslabels = read_inslabels(label_file)
    tags = np.fromfile(tag_file, dtype = np.bool)
    label_vec = np.zeros((H*W), dtype = np.uint32)
    label_vec[tags] = inslabels
    image_vec = SEM_COLOR[label_vec%23]
    insimage = np.reshape(image_vec, (H,W,3))
    return insimage

### code adapted from SemanticKitti
from torch.utils.data import Dataset
import torch
EXTENSIONS_SCAN = ['.bin']
EXTENSIONS_EDGE = ['.png']
EXTENSIONS_LABEL = ['.label']
EXTENSIONS_TAG = ['.tag']

def is_scan(filename):
  return any(filename.endswith(ext) for ext in EXTENSIONS_SCAN)

def is_edge(filename):
  return any(filename.endswith(ext) for ext in EXTENSIONS_EDGE)

def is_label(filename):
  return any(filename.endswith(ext) for ext in EXTENSIONS_LABEL)

def is_tag(filename):
  return any(filename.endswith(ext) for ext in EXTENSIONS_TAG)

class LaserScanPOSS:
  """Class that contains LaserScan with x,y,z,r"""
  EXTENSIONS_SCAN = ['.bin']

  def __init__(self, resized_H=40, resized_W=1024, poss2kitti=None):
    self.resized_H = H
    self.resized_W = W
    self.proj_W = 1800  # width of range image
    self.proj_H = 40  # height of range image
    self.poss2kitti = poss2kitti
    self.reset()

  def reset(self):
    """ Reset scan members. """
    self.points = np.zeros((0, 3), dtype=np.float32)        # [m, 3]: x, y, z
    self.remissions = np.zeros((0, 1), dtype=np.float32)    # [m ,1]: remission

    # projected range image - [H,W] range (-1 is no data)
    self.proj_range = np.full((self.proj_H, self.proj_W), -1,
                              dtype=np.float32)

    # unprojected range (list of depths for each point)
    self.unproj_range = np.zeros((0, 1), dtype=np.float32)

    # projected point cloud xyz - [H,W,3] xyz coord (-1 is no data)
    self.proj_xyz = np.full((self.proj_H, self.proj_W, 3), -1,
                            dtype=np.float32)

    # projected remission - [H,W] intensity (-1 is no data)
    self.proj_remission = np.full((self.proj_H, self.proj_W), -1,
                                  dtype=np.float32)

    # projected index (for each pixel, what I am in the pointcloud)
    # [H,W] index (-1 is no data)
    self.proj_idx = np.full((self.proj_H, self.proj_W), -1,
                            dtype=np.int32)

    # for each point, where it is in the range image
    self.proj_x = np.zeros((0, 1), dtype=np.int32)        # [m, 1]: x
    self.proj_y = np.zeros((0, 1), dtype=np.int32)        # [m, 1]: y

    # mask containing for each pixel, if it contains a point or not
    self.proj_mask = np.zeros((self.proj_H, self.proj_W),
                              dtype=np.int32)       # [H,W] mask

  def size(self):
    """ Return the size of the point cloud. """
    return self.points.shape[0]

  def __len__(self):
    return self.size()

  def open_scan(self, bin_file, label_file, tag_file=None):
    """ Open raw scan and fill in attributes
    """
    # reset just in case there was an open structure
    self.reset()

    # # check filename is string
    # if not isinstance(filename, str):
    #   raise TypeError("Filename should be string type, "
    #                   "but was {type}".format(type=str(type(filename))))
    #
    # # check extension is a laserscan
    # if not any(filename.endswith(ext) for ext in self.EXTENSIONS_SCAN):
    #   raise RuntimeError("Filename extension is not valid scan file.")

    scan = read_points(bin_file)                            ## points after tags mask
    semlabels = read_semlabels(label_file)
    # mapping
    if self.poss2kitti:                     ## for labels, not for predictions
        semlabels = self.map(semlabels, self.poss2kitti)
    self.sem_label = semlabels

    # put in attribute
    points = scan[:, 0:3]    # get xyz
    remissions = scan[:, 3]  # get remission

    # mask = None
    # mask1 = abs(scan[:, 0]) < 80
    # mask2 = abs(scan[:, 1]) < 80
    # mask = mask1 * mask2
    # points[:, 0] = points[:, 0] *mask
    # points[:, 1] = points[:, 1] *mask
    # points[:, 2] = points[:, 2] *mask
    # remissions = remissions*mask
    # if self.poss2kitti:
    #     semlabels = semlabels*mask
    #     self.sem_label = semlabels
    # # print(mask1.shape, mask2.shape, mask.shape, scan.shape, points.shape)


    # points[:, 0] = points[:, 0] *0.4 # worse
    # points[:, 1] = points[:, 1] *0.4 # worse
    # points[:, 2] = points[:, 2]

    remissions = remissions / 255.

    # print("AFTER: WE ARE HERE")
    # for i in range(3):
    #   print("max={:.2f}, min={:.2f}, abs_mean={:.2f}, mean={:.2f}"
    #         .format(points[:, i].max(), points[:, i].min(), np.abs(points[:, i]).mean(), points[:, i].mean()))
    self.set_points(points, remissions)

    if tag_file:
        tags = np.fromfile(tag_file, dtype = np.bool)           ## all points in a 2D image, e.g. H*W

        # projection
        # dis = np.linalg.norm(points, axis=1)
        dis = np.linalg.norm(points, axis = 1)        ## * 5 for better visualization (actually4 > 3,5)
        # dis = np.minimum(dis, 255.)                      ## *5 + 255. = focus on range=0~50m
        dis = dis.astype(np.float32)
        self.unproj_range = np.copy(dis)
        # dis_vec = np.zeros((H*W), dtype = np.uint8)
        dis_vec = np.full((H*W), -1, dtype=np.float32)
        dis_vec[tags] = dis
        self.proj_range = np.reshape(dis_vec, (H,W))
        # self.proj_range = cv.resize(self.proj_range,(self.resized_W, self.resized_H))       # fixme

        # points_vec = np.zeros((H*W), dtype = np.float32)
        points_vec = np.full((H*W, 3), -1, dtype=np.float32)
        points_vec[tags] = self.points
        self.proj_xyz = np.reshape(points_vec, (H, W, 3))
        # self.proj_xyz = cv.resize(self.proj_xyz,(self.resized_W, self.resized_H))# fixme

        # remission_vec = np.zeros((H*W), dtype = np.float32)
        remission_vec = np.full((H * W), -1, dtype=np.float32)
        remission_vec[tags] = self.remissions
        self.proj_remission = np.reshape(remission_vec, (H,W))
        # self.proj_remission = cv.resize(self.proj_remission,(self.resized_W, self.resized_H))# fixme

        label_vec = np.zeros((H*W), dtype = np.uint32)
        label_vec[tags] = semlabels
        self.proj_sem_label = np.reshape(label_vec, (H, W))
        # self.proj_sem_label = cv.resize(self.proj_sem_label,(self.resized_W, self.resized_H))

        self.proj_mask = np.reshape(tags, (H, W)).astype(np.int32)
        self.proj_y, self.proj_x = self.proj_mask.nonzero()

  def set_points(self, points, remissions=None):
    """ Set scan attributes (instead of opening from file)
    """
    # reset just in case there was an open structure
    self.reset()

    # check scan makes sense
    if not isinstance(points, np.ndarray):
      raise TypeError("Scan should be numpy array")

    # check remission makes sense
    if remissions is not None and not isinstance(remissions, np.ndarray):
      raise TypeError("Remissions should be numpy array")

    # put in attribute
    self.points = points    # get xyz
    if remissions is not None:
      self.remissions = remissions  # get remission
    else:
      self.remissions = np.zeros((points.shape[0]), dtype=np.float32)

  @staticmethod
  def map(label, mapdict):  ## label: cls_id in content  mapdict: learning_map  fixme?
    # put label from original values to xentropy
    # or vice-versa, depending on dictionary values
    # make learning map a lookup table
    maxkey = 0  ## 0->1->10->11->...->259
    for key, data in mapdict.items():
      if isinstance(data, list):  ## never
        nel = len(data)
      else:
        nel = 1
      if key > maxkey:
        maxkey = key
    # +100 hack making lut bigger just in case there are unknown labels
    if nel > 1:  ## never
      lut = np.zeros((maxkey + 100, nel), dtype=np.int32)
    else:
      lut = np.zeros((maxkey + 100), dtype=np.int32)
    for key, data in mapdict.items():
      try:
        lut[key] = data
      except IndexError:
        print("Wrong key ", key)
    # do the mapping
    return lut[label]

class SemanticPOSS(Dataset):
    def __init__(self, root,  # directory where data is
                 sequences,  # sequences for this data (e.g. [1,3,4,6])
                 labels,  # label dict: (e.g 10: "car")
                 color_map,  # colors dict bgr (e.g 10: [255, 0, 0])
                 learning_map,  # classes to learn (0 to N-1 for xentropy)
                 learning_map_inv,  # inverse of previous (recover labels)
                 sensor,  # sensor to parse scans from
                 max_points=150000,  # max number of points present in dataset
                 gt=True, skip=0, poss2kitti=None, max_iters=None):  # send ground truth?     ###
        # save deats
        self.root = os.path.join(root, "sequences")
        self.sequences = sequences
        self.labels = labels
        self.color_map = color_map
        self.learning_map = learning_map
        self.learning_map_inv = learning_map_inv
        self.sensor = sensor
        self.sensor_img_H = sensor["img_prop"]["height"]
        self.sensor_img_W = sensor["img_prop"]["width"]
        self.sensor_img_means = torch.tensor(sensor["img_means"],
                                             dtype=torch.float)
        self.sensor_img_stds = torch.tensor(sensor["img_stds"],
                                            dtype=torch.float)
        self.sensor_fov_up = sensor["fov_up"]
        self.sensor_fov_down = sensor["fov_down"]
        self.max_points = max_points
        self.gt = gt
        ###
        self.poss2kitti = poss2kitti
        self.class_names = [self.labels[value] for value in self.learning_map_inv.values()]
        self.max_iters = max_iters

        # get number of classes (can't be len(self.learning_map) because there
        # are multiple repeated entries, so the number that matters is how many
        # there are for the xentropy)
        self.nclasses = len(self.learning_map_inv)

        # sanity checks

        # make sure directory exists
        if os.path.isdir(self.root):
            print("Sequences folder exists! Using sequences from %s" % self.root)
        else:
            raise ValueError("Sequences folder doesn't exist! Exiting...")

        # make sure labels is a dict
        assert (isinstance(self.labels, dict))

        # make sure color_map is a dict
        assert (isinstance(self.color_map, dict))

        # make sure learning_map is a dict
        assert (isinstance(self.learning_map, dict))

        # make sure sequences is a list
        assert (isinstance(self.sequences, list))

        # placeholder for filenames
        self.scan_files = []
        # self.edge_files = []
        self.label_files = []
        self.tag_files = []             ###

        # fill in with names, checking that all sequences are complete
        for seq in self.sequences:
            # to string
            seq = '{0:02d}'.format(int(seq))

            print("parsing seq {}".format(seq))

            # get paths for each
            scan_path = os.path.join(self.root, seq, "velodyne")
            # edge_path = os.path.join(self.root, seq, "edges")
            label_path = os.path.join(self.root, seq, "labels")
            tag_path = os.path.join(self.root, seq, "tag")

            # get files
            scan_files = [os.path.join(dp, f) for dp, dn, fn in os.walk(
                os.path.expanduser(scan_path)) for f in fn if is_scan(f)]
            # edge_files = [os.path.join(dp, f) for dp, dn, fn in os.walk(
            #     os.path.expanduser(edge_path)) for f in fn if is_edge(f)]
            label_files = [os.path.join(dp, f) for dp, dn, fn in os.walk(
                os.path.expanduser(label_path)) for f in fn if is_label(f)]
            tag_files = [os.path.join(dp, f) for dp, dn, fn in os.walk(
                os.path.expanduser(tag_path)) for f in fn if is_tag(f)]

            # assert(len(scan_files) == len(edge_files))

            # check all scans have labels
            if self.gt:
                assert (len(scan_files) == len(label_files))


            # extend list
            self.scan_files.extend(scan_files)
            # self.edge_files.extend(edge_files)
            self.label_files.extend(label_files)
            self.tag_files.extend(tag_files)

        # sort for correspondance
        self.scan_files.sort()
        # self.edge_files.sort()
        self.label_files.sort()
        self.tag_files.sort()

        ###
        if self.max_iters:
            self.scan_files = self.scan_files  * int(np.ceil(float(self.max_iters) / len(self.scan_files)))
            self.label_files = self.label_files  * int(np.ceil(float(self.max_iters) / len(self.label_files)))
            self.tag_files = self.tag_files  * int(np.ceil(float(self.max_iters) / len(self.tag_files)))


        if skip != 0:
            self.scan_files = self.scan_files[::skip]
            # self.edge_files = self.edge_files[::skip]
            self.label_files = self.label_files[::skip]
            self.tag_files = self.tag_files[::skip]

        print("Using {} scans from sequences {}".format(len(self.scan_files),
                                                        self.sequences))

    def __getitem__(self, index):
        # get item in tensor shape
        scan_file = self.scan_files[index]
        # edge_file = self.edge_files[index]
        tag_file = self.tag_files[index]
        if self.gt:
            label_file = self.label_files[index]

        # open a semantic laserscan
        scan = LaserScanPOSS(resized_H=self.sensor_img_H, resized_W=self.sensor_img_W, poss2kitti=self.poss2kitti)      ###

        # open and obtain scan
        scan.open_scan(scan_file, label_file, tag_file)

        scan.edge = 1  ###
        if self.gt:
            # scan.open_label(label_file)
            # map unused classes to used classes (also for projection)
            scan.sem_label = self.map(scan.sem_label, self.learning_map)
            scan.proj_sem_label = self.map(scan.proj_sem_label, self.learning_map)

        # make a tensor of the uncompressed data (with the max num points)
        unproj_n_points = scan.points.shape[0]
        unproj_xyz = torch.full((self.max_points, 3), -1.0, dtype=torch.float)
        unproj_xyz[:unproj_n_points] = torch.from_numpy(scan.points)
        unproj_range = torch.full([self.max_points], -1.0, dtype=torch.float)
        unproj_range[:unproj_n_points] = torch.from_numpy(scan.unproj_range)
        unproj_remissions = torch.full([self.max_points], -1.0, dtype=torch.float)
        unproj_remissions[:unproj_n_points] = torch.from_numpy(scan.remissions)
        if self.gt:
            unproj_labels = torch.full([self.max_points], -1.0, dtype=torch.int32)
            unproj_labels[:unproj_n_points] = torch.from_numpy(scan.sem_label)
        else:
            unproj_labels = []

        # get points and labels
        proj_range = torch.from_numpy(scan.proj_range).clone()
        proj_xyz = torch.from_numpy(scan.proj_xyz).clone()
        proj_remission = torch.from_numpy(scan.proj_remission).clone()
        proj_mask = torch.from_numpy(scan.proj_mask)
        if self.gt:
            proj_labels = torch.from_numpy(scan.proj_sem_label).clone()
            proj_labels = proj_labels * proj_mask
        else:
            proj_labels = []
        proj_x = torch.full([self.max_points], -1, dtype=torch.long)
        proj_x[:unproj_n_points] = torch.from_numpy(scan.proj_x)
        proj_y = torch.full([self.max_points], -1, dtype=torch.long)
        proj_y[:unproj_n_points] = torch.from_numpy(scan.proj_y)
        proj = torch.cat([proj_range.unsqueeze(0).clone(),
                          proj_xyz.clone().permute(2, 0, 1),
                          proj_remission.unsqueeze(0).clone()])
        proj = (proj - self.sensor_img_means[:, None, None]) / self.sensor_img_stds[:, None, None]
        proj = proj * proj_mask.float()

        # get name and sequence
        path_norm = os.path.normpath(scan_file)
        path_split = path_norm.split(os.sep)
        path_seq = path_split[-3]
        path_name = path_split[-1].replace(".bin", ".label")
        # print("path_norm: ", path_norm)
        # print("path_seq", path_seq)
        # print("path_name", path_name)

        # return
        return proj, proj_mask, proj_labels, unproj_labels, path_seq, path_name, proj_x, proj_y, proj_range, unproj_range, proj_xyz, unproj_xyz, proj_remission, unproj_remissions, unproj_n_points, scan.edge

    def __len__(self):

        return len(self.scan_files)

    @staticmethod
    def map(label, mapdict):
        # put label from original values to xentropy
        # or vice-versa, depending on dictionary values
        # make learning map a lookup table
        maxkey = 0
        for key, data in mapdict.items():
            if isinstance(data, list):
                nel = len(data)
            else:
                nel = 1
            if key > maxkey:
                maxkey = key
        # +100 hack making lut bigger just in case there are unknown labels
        if nel > 1:
            lut = np.zeros((maxkey + 100, nel), dtype=np.int32)
        else:
            lut = np.zeros((maxkey + 100), dtype=np.int32)
        for key, data in mapdict.items():
            try:
                lut[key] = data
            except IndexError:
                print("Wrong key ", key)
        # do the mapping
        return lut[label]

def get_video():
    resized_W = 1024
    resized_H = 40#128

    data_path = '/media/datasets/yij/DA_seg/SemanticPOSS/dataset/sequences/00/'
    videoWriter = cv.VideoWriter('/media/datasets/yij/DA_seg/SemanticPOSS/dataset/00.avi',cv.VideoWriter_fourcc('M','J','P','G'),10,(resized_W,resized_H*2+64),True)    ## todo


    for bin_file,label_file,tag_file in zip(sorted(os.listdir(data_path+'velodyne/')),
            sorted(os.listdir(data_path+'labels/')),sorted(os.listdir(data_path+'tag/'))):
        rangeimage = get_rangeimage(data_path+'velodyne/'+bin_file, data_path+'tag/'+tag_file)
        rangeimage = cv.resize(rangeimage,(resized_W, resized_H))

        whiteimage = np.ones((64,resized_W,3),dtype = np.uint8)*255

        semimage = get_semimage(data_path+'labels/'+label_file, data_path+'tag/'+tag_file)
        semimage2 = cv.resize(semimage,(resized_W, resized_H))

        mergeimage = np.concatenate((rangeimage,whiteimage,semimage2),axis = 0)

        videoWriter.write(mergeimage)           ## todo
    videoWriter.release()                       ## todo


if __name__=='__main__': # an example for generating video of sequence 00
    get_video()




