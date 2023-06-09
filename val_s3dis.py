import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import torch
import time
import numpy as np
from utils import iou_s3dis
from model.backbone import AIA
from MinkowskiEngine import SparseTensor
from utils.util import logging, bar
from s3dis.s3dis import load_segment, s3disEvaluate, collate_fn_evaluate
import glob
from torch.utils.data import DataLoader
from scipy.sparse import coo_matrix

def infer(model, dataloader, normalize_color=True, device=None):
    with torch.no_grad():
        model.eval()
        start = time.time()
        output_dict = {}  # save voting result, pcd, colors and labels
        for i, batch in enumerate(dataloader):
            coords, input, target, remap_index, file_names,coords_raw,superpoint = batch
            xyz = coords
            torch.cuda.empty_cache()
            # For some networks, making the network invariant to even, odd coords is important
            coords[:, 1:] += (torch.rand(3) * 100).type_as(coords)
            # Preprocess input
            if normalize_color:
                input[:, :3] = input[:, :3] / 255. - 0.5
            sinput = SparseTensor(input.cuda(), coords.cuda())
            # Feed forward
            preds_3d = model.forward_eval_s3dis(sinput,coords_raw,superpoint,target)
            batch_idxs = coords[:, 0].numpy()  # [N]
            torch.cuda.empty_cache()
            for j, b in enumerate(np.unique(batch_idxs)):
                idxs = np.where(batch_idxs == b)[0]
                remap_idx = remap_index[j]
                logits = preds_3d[idxs][remap_idx]  # remap to original shape
                target_single = target[idxs][remap_idx]  # remap to original shape
                xyz_j = xyz[idxs][remap_idx][:, 1:]
                rgb_j = input[idxs][remap_idx]
                # print(xyz_j.shape, rgb_j.shape, logits.shape)
                output_dict[file_names[j]] = [logits.cpu().numpy(), target_single.numpy().astype(int), xyz_j, rgb_j]
            bar(f'time {time.time() - start:.1f}', i + 1, len(dataloader))
            torch.cuda.empty_cache()
        print()
    return output_dict
def vote_by_segment_uncertainty_topk(predictions, segments, uncertainty=None, topk=-1, device=None):
    """Reduce predictions by segments and broadcast to each point in the segments.
    :param predictions: [N, C], ndarray or torch.Tensor, prediction of model (logits or softmax).
    :param segments: [N], ndarray or torch.Tensor, segment id for each point.
    :param uncertainty: [N], uncertainty of each point.
    :param topk: the topk point, when topk=-1, all the point will include.
    :param device:
    :return refine: [N, C] refine predictions.
    """
    if isinstance(predictions, np.ndarray):
        predictions = torch.tensor(predictions)
    if isinstance(segments, np.ndarray):
        segments = torch.tensor(segments)
    if uncertainty is not None and isinstance(uncertainty, np.ndarray):
        uncertainty = torch.tensor(uncertainty)

    predictions = predictions.cuda()
    segments = segments.cuda()
    if uncertainty is not None:
        uncertainty = uncertainty.cuda()

    refine = torch.zeros(predictions.shape).cuda()
    for seg_id in torch.unique(segments):
        idx = torch.where(segments == seg_id)[0]
        if topk > 0 and uncertainty is not None:
            u = -uncertainty[idx]
            min_k = min(topk, len(idx))
            filter_idx = u.topk(min_k)[1]  # get `k` most certain point to represent the segment
            res = torch.mean(predictions[idx][filter_idx], dim=0)
        else:
            res = torch.mean(predictions[idx], dim=0)
        refine[idx] = res
    return refine


def compute_miou(result, save_log=False):
    pred_list, label_list = [], []
    for f in result.keys():
        pred_list.append(result[f][0])
        label_list.append(result[f][1])
    if len(pred_list[0].shape) >1:
        miou = iou_s3dis.evaluate(np.vstack(pred_list).argmax(1), np.hstack(label_list), save_log=save_log)
    else:
        miou = iou_s3dis.evaluate(np.hstack(pred_list), np.hstack(label_list), save_log=save_log)
    logging(f'miou {miou * 100:.2f}\n', save_log=save_log)
    return miou
def align_superpoint_label(labels: torch.Tensor,
                           superpoint: torch.Tensor,
                           num_label: int=20,
                           ignore_label: int=-100):
    r"""refine semantic segmentation by superpoint

    Args:
        labels (torch.Tensor, [N]): semantic label of points
        superpoint (torch.Tensor, [N]): superpoint cluster id of points
        num_label (int): number of valid label categories
        ignore_label (int): the ignore label id

    Returns:
        label: (torch.Tensor, [num_superpoint]): superpoint's label
        label_scores: (torch.Tensor, [num_superpoint, num_label + 1]): superpoint's label scores
    """
    row = superpoint# superpoint has been compression
    col = labels
    col[col < 0] = num_label
    data = np.ones(len(superpoint))
    shape = (len(np.unique(row)), num_label + 1)
    label_map = coo_matrix((data, (row, col)), shape=shape).toarray()  # [num_superpoint, num_label + 1]
    label = np.argmax(label_map, axis=1) # [num_superpoint]
    label[label == num_label] = ignore_label # ignore_label
    label_scores = label_map.max(1) / label_map.sum(axis=1) # [num_superpoint, num_label + 1]

    return label, label_scores


if __name__ == '__main__':
    

    run_reps = 8
    batchsize = 4

    phase = 'val'
    #DEVICE = torch.device('cuda')

    data_root = ''
    segment_root = data_root
    home_path1 = os.path.dirname(__file__)
    model_path1 = home_path1+'/pth/s3dis.pth'
    voxel_size = 0.02
    num_workers = 0
    normalize_color = True

    augment_data = True
    elastic_distortion = False
    ELASTIC_DISTORT_PARAMS = ((0.2, 0.4), (0.8, 1.6))
    random_scale = True
    SCALE_AUGMENTATION_BOUND = (0.9, 1.1)
    random_rotation = True
    ROTATION_AUGMENTATION_BOUND = ((-np.pi / 64, np.pi / 64), (-np.pi / 64, np.pi / 64), (-np.pi, np.pi))
    random_flip = True
    ROTATION_AXIS = 'z'
    chromaticautocontrast = True
    chromatictranslation = True
    data_aug_color_trans_ratio = 0.1
    chromaticjitter = True
    data_aug_color_jitter_std = 0.05

    unaugment_loader = DataLoader(
        dataset=s3disEvaluate(
            phase, save_log=False, data_root=data_root, voxel_size=voxel_size,
            ignore_label=-100, augment_data=False),
        batch_size=batchsize,
        num_workers=num_workers,
        collate_fn=collate_fn_evaluate,
        shuffle=False)

    loader = DataLoader(
        dataset=s3disEvaluate(
            phase, save_log=False, data_root=data_root, voxel_size=voxel_size,
            ignore_label=-100, augment_data=augment_data, elastic_distortion=elastic_distortion,
            ELASTIC_DISTORT_PARAMS=ELASTIC_DISTORT_PARAMS,
            random_scale=random_scale, SCALE_AUGMENTATION_BOUND=SCALE_AUGMENTATION_BOUND,
            random_rotation=random_rotation,
            ROTATION_AUGMENTATION_BOUND=ROTATION_AUGMENTATION_BOUND,
            random_flip=random_flip, ROTATION_AXIS=ROTATION_AXIS,
            chromaticautocontrast=chromaticautocontrast,
            chromatictranslation=chromatictranslation, data_aug_color_trans_ratio=data_aug_color_trans_ratio,
            chromaticjitter=chromaticjitter, data_aug_color_jitter_std=data_aug_color_jitter_std),
        batch_size=batchsize,
        num_workers=num_workers,
        collate_fn=collate_fn_evaluate,
        shuffle=False)

    res = None

    model = AIA(in_channels=3,features=64,out_channels=13,num_heads=2,feat_size=128).cuda()
    state_dict = torch.load(model_path1)
    model.load_state_dict(state_dict)

    #unaugment
    for i in range(1):
        print(f'unaugment use {model_path1} reps {i}')
        out = infer(model, unaugment_loader, True)
        if res is None:
            res = out
        else:
            for k in out.keys():
                res[k][0] += out[k][0]
        compute_miou(res)

    #augment
    for i in range(20):
        print(f'augment use {model_path1} reps {i}')
        out = infer(model, loader, True)
        if res is None:
            res = out
        else:
            for k in out.keys():
                res[k][0] += out[k][0]
        compute_miou(res)

    logging(f'vote segment', save_log=False)
    #Compute pseudo label miou
    for i, f in enumerate(res.keys()):
        logits = res[f][0]#.argmax(1)
        path_to_segment = glob.glob(segment_root+'/*')
        if len(path_to_segment) > 0:
            segment = torch.tensor(np.int32(torch.load(data_root+'/'+f+'.pth')[-1]))
            res[f][0] = vote_by_segment_uncertainty_topk(logits, segment, None, topk=-1,
                                                      ).cpu().numpy()
        else:
            raise ValueError('No file name {}/{}*'.format(segment_root, f))
        bar(f'reduce segment', i + 1, len(res.keys()))
    print()
    compute_miou(res)
