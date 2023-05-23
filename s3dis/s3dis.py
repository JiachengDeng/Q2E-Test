import torch
import glob
import numpy as np
from utils.util import logging, bar
import MinkowskiEngine as ME
import scannet.transforms as t
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler
import json
import torch.nn.functional as F
from scipy.sparse import coo_matrix

def load_segment(path):
    f = open(path, 'r')
    seg = json.load(f)['segIndices']  # [N]
    return np.array(seg)


def load_data(path):
    """Load original data
    :return coords: [N, 3].
    :return feats: [N, 3], RGB colors(0~255).
    :return labels: [N], 0~19, -100 indicates invalid label.
    """
    coords, feats, labels = torch.load(path)[:3]
    return coords, feats, labels


class InfSampler(Sampler):
    """Samples elements randomly, without replacement.
      Arguments:
          data_source (Dataset): dataset to sample from
      """

    def __init__(self, data_source, shuffle=False):
        self.data_source = data_source
        self.shuffle = shuffle
        self.reset_permutation()

    def reset_permutation(self):
        perm = len(self.data_source)
        if self.shuffle:
            perm = torch.randperm(perm)
        self._perm = perm.tolist()

    def __iter__(self):
        return self

    def __next__(self):
        if len(self._perm) == 0:
            self.reset_permutation()
        return self._perm.pop()

    def __len__(self):
        return len(self.data_source)

    next = __next__  # Python 2 compatibility


class ProcessCoords(object):
    def __init__(
            self,
            voxel_size=0.02, ignore_label=-100,
            elastic_distortion=False, ELASTIC_DISTORT_PARAMS=((0.2, 0.4), (0.8, 1.6)),
            random_scale=True, SCALE_AUGMENTATION_BOUND=(0.9, 1.1),
            random_rotation=False,
            ROTATION_AUGMENTATION_BOUND=((-np.pi / 64, np.pi / 64), (-np.pi / 64, np.pi / 64), (-np.pi, np.pi)),
            random_flip=True, ROTATION_AXIS='z'
    ):
        input_transforms = []
        if elastic_distortion:
            input_transforms.append(t.ElasticDistortion(ELASTIC_DISTORT_PARAMS))
        if random_rotation:
            input_transforms.append(t.RandomRotation(ROTATION_AUGMENTATION_BOUND))
        if random_flip:
            input_transforms.append(t.RandomHorizontalFlip(ROTATION_AXIS, False))
        self.input_transforms = t.Compose(input_transforms)
        #self.voxelizer = t.Voxelize(voxel_size, random_scale, SCALE_AUGMENTATION_BOUND, ignore_label=ignore_label)

    def __call__(self, coords, colors, labels):
        coords, colors, labels = self.input_transforms(coords, colors, labels)
        #coords, colors, labels, remap_index = self.voxelizer(coords, colors, labels)
        return coords, colors, labels
    
class ProcessCoordseval(object):
    def __init__(
            self,
            voxel_size=0.02, ignore_label=-100,
            elastic_distortion=False, ELASTIC_DISTORT_PARAMS=((0.2, 0.4), (0.8, 1.6)),
            random_scale=True, SCALE_AUGMENTATION_BOUND=(0.9, 1.1),
            random_rotation=False,
            ROTATION_AUGMENTATION_BOUND=((-np.pi / 64, np.pi / 64), (-np.pi / 64, np.pi / 64), (-np.pi, np.pi)),
            random_flip=True, ROTATION_AXIS='z'
    ):
        input_transforms = []
        if elastic_distortion:
            input_transforms.append(t.ElasticDistortion(ELASTIC_DISTORT_PARAMS))
        if random_rotation:
            input_transforms.append(t.RandomRotation(ROTATION_AUGMENTATION_BOUND))
        if random_flip:
            input_transforms.append(t.RandomHorizontalFlip(ROTATION_AXIS, False))
        self.input_transforms = t.Compose(input_transforms)
        self.voxelizer = t.Voxelize(voxel_size, random_scale, SCALE_AUGMENTATION_BOUND, ignore_label=ignore_label)

    def __call__(self, coords, colors, labels):
        coords, colors, labels = self.input_transforms(coords, colors, labels)
        coords, colors, labels,coords_raw, remap_index = self.voxelizer(coords, colors, labels)
        return coords, colors, labels,coords_raw, remap_index

class ProcessColors(object):
    def __init__(
            self,
            chromaticautocontrast=True,
            chromatictranslation=True, data_aug_color_trans_ratio=0.1,
            chromaticjitter=True, data_aug_color_jitter_std=0.05
    ):
        input_transforms = []
        if chromaticautocontrast:
            input_transforms.append(t.ChromaticAutoContrast())
        if chromatictranslation:
            input_transforms.append(t.ChromaticTranslation(data_aug_color_trans_ratio))
        if chromaticjitter:
            input_transforms.append(t.ChromaticJitter(data_aug_color_jitter_std))
        self.input_transforms = t.Compose(input_transforms)

    def __call__(self, coords, colors, labels):
        coords, colors, labels = self.input_transforms(coords, colors, labels)
        return coords, colors, labels


#################
# evaluate data #
#################
class s3disEvaluate(Dataset):
    def __init__(
            self, phase='val', save_log=True, data_root='path2data',
            voxel_size=0.02, ignore_label=-100, augment_data=False,
            elastic_distortion=True, ELASTIC_DISTORT_PARAMS=((0.2, 0.4), (0.8, 1.6)),
            random_scale=True, SCALE_AUGMENTATION_BOUND=(0.9, 1.1),
            random_rotation=True,
            ROTATION_AUGMENTATION_BOUND=((-np.pi / 64, np.pi / 64), (-np.pi / 64, np.pi / 64), (-np.pi, np.pi)),
            random_flip=True, ROTATION_AXIS='z',
            chromaticautocontrast=True,
            chromatictranslation=True, data_aug_color_trans_ratio=0.1,
            chromaticjitter=True, data_aug_color_jitter_std=0.05,
            **kwargs
    ):
        logging(' * ' * 10 + 'Initialise s3dis Dataset' + ' * ' * 10, save_log=save_log)
        # load files path
        self.data_paths, self.file_names = [], []
        #f = open(SPLIT_TXT_PATH[phase], 'r')
        self.data_paths= sorted(glob.glob(f'{data_root}/*'))
        for file in self.data_paths:
            # file name, e.g. scene0191_00
    
            self.file_names.append(file.split('/')[-1][:-4])
        self.file_names = sorted(self.file_names)

        if not augment_data:
            elastic_distortion = False
            random_scale = False
            random_rotation = False
            random_flip = False
            chromaticautocontrast = False
            chromatictranslation = False
            chromaticjitter = False

        log = f'phase {phase}\n' \
              f'data_root {data_root}\n' \
              f'voxel_size {voxel_size}\n' \
              f'augment_data {augment_data}\n' \
              f'elastic_distortion {elastic_distortion} ELASTIC_DISTORT_PARAMS {ELASTIC_DISTORT_PARAMS}\n' \
              f'random_scale {random_scale} SCALE_AUGMENTATION_BOUND {SCALE_AUGMENTATION_BOUND}\n' \
              f'random_rotation {random_rotation} ROTATION_AUGMENTATION_BOUND {ROTATION_AUGMENTATION_BOUND}\n' \
              f'random_flip {random_flip} ROTATION_AXIS {ROTATION_AXIS}\n' \
              f'chromaticautocontrast {chromaticautocontrast}\n' \
              f'chromatictranslation {chromatictranslation} data_aug_color_trans_ratio {data_aug_color_trans_ratio}\n' \
              f'chromaticjitter {chromaticjitter} data_aug_color_jitter_std {data_aug_color_jitter_std}\n'
        logging(log, save_log=save_log)

        # coords augmentation
        self.process_coords = ProcessCoordseval(
            voxel_size=voxel_size, ignore_label=ignore_label,
            elastic_distortion=elastic_distortion, ELASTIC_DISTORT_PARAMS=ELASTIC_DISTORT_PARAMS,
            random_scale=random_scale, SCALE_AUGMENTATION_BOUND=SCALE_AUGMENTATION_BOUND,
            random_rotation=random_rotation, ROTATION_AUGMENTATION_BOUND=ROTATION_AUGMENTATION_BOUND,
            random_flip=random_flip, ROTATION_AXIS=ROTATION_AXIS
        )
        # colors augmentation
        self.process_colors = ProcessColors(
            chromaticautocontrast=chromaticautocontrast,
            chromatictranslation=chromatictranslation, data_aug_color_trans_ratio=data_aug_color_trans_ratio,
            chromaticjitter=chromaticjitter, data_aug_color_jitter_std=data_aug_color_jitter_std
        )

    def __getitem__(self, index):
        points = torch.load(self.data_paths[index])
        coords, colors, labels,segment  = (
                points[0],
                points[1],
                points[2],
                points[-1],
            )
        # process coords
        labels = np.concatenate([segment[:,None],labels[:,None]],axis=-1)
        coords, colors, labels, coords_raw,remap_index = self.process_coords(coords, colors, labels)
        superpoint = labels[:,0]
        labels = labels[:,1]
        # process colors
        coords, colors, labels = self.process_colors(coords, colors, labels)

        #labels = self.remapper[np.int32(labels)]
        return tuple([coords, colors, labels, remap_index, self.file_names[index],torch.tensor(coords_raw).type(torch.float32),torch.tensor(superpoint).int()])

    def __len__(self):
        return len(self.data_paths)


def collate_fn_evaluate(list_data):
    """Generates collate function for coords, feats, labels, remap_idx, filename.
    """
    coords, feats, labels, remap_idxs, file_names,coords_raw,superpoint = list(zip(*list_data))
    coords_batch, feats_batch, labels_batch, remap_batch, file_batch = [], [], [], [], []
    coords_batch_raw = []
    for batch_id, _ in enumerate(coords):
        coords_batch.append(torch.from_numpy(coords[batch_id]).int())
        feats_batch.append(torch.from_numpy(feats[batch_id]))
        labels_batch.append(torch.from_numpy(labels[batch_id]).int())
        remap_batch.append(remap_idxs[batch_id])
        file_batch.append(file_names[batch_id])
        coords_batch_raw.append(coords_raw[batch_id].type(torch.float32))
    # Concatenate all lists
    coords_batch, feats_batch, labels_batch = ME.utils.sparse_collate(coords_batch, feats_batch, labels_batch)
    _, coords_batch_raw = ME.utils.sparse_collate(coords_batch_raw, coords_batch_raw)
    return coords_batch, feats_batch.float(), labels_batch, remap_batch, file_batch,coords_batch_raw,superpoint


def get_evaluate_loader(
        cfg, data_root, phase, batchsize=1, num_workers=1, augment_data=False, shuffle=False, save_log=False
):
    dataset = s3disEvaluate(
        phase, save_log=save_log, data_root=data_root, voxel_size=cfg['DATA']['voxel_size'],
        ignore_label=cfg['DATA']['ignore_label'], augment_data=augment_data, **cfg['AUGMENTATION']
    )

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batchsize,
        num_workers=num_workers,
        collate_fn=collate_fn_evaluate,
        shuffle=shuffle
    )
    return dataloader


