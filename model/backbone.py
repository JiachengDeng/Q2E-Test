import torch
import torch.nn as nn
import MinkowskiEngine.MinkowskiOps as me
import numpy as np
from torch.nn import functional as F
from model.minkunet import MinkUNet34C
from model.dm_module import DMModule
from third_party.pointnet2.pointnet2_utils import furthest_point_sample
from torch.cuda.amp import autocast
from torch_scatter import scatter_mean
from model.position_embedding import PositionEmbeddingCoordsSine
from collections import OrderedDict
class AIA(nn.Module):
    def __init__(self,in_channels=3, features=512, out_channels=20,num_heads=4,feat_size=512
                 ):
        super().__init__()
        self.backbone = MinkUNet34C(in_channels=in_channels, out_channels=features)
        self.cls = nn.Linear(features,out_channels)
        self.cls1 = nn.Sequential(OrderedDict([
            ("bn1", nn.LayerNorm(features)),
            ("layer1", nn.Linear(features, 1)),
        ]))
        self.attention = nn.ModuleList()
        self.num_heads = num_heads
        for i in range(2):
          self.attention.append(
                      DMLayer(
                          d_model=features,
                          nhead=self.num_heads,
                          dropout=0.1,
                          normalize_before=True,
                          feat_size = feat_size
                      )
                  )

        self.pos_enc = PositionEmbeddingCoordsSine(pos_type="fourier",
                                                       d_pos=features,
                                                       gauss_scale=0.1,
                                                       normalize=True)

    def get_pos_encs(self, coords):
        pos_encodings_pcd = []

        for i in range(len(coords)):
            pos_encodings_pcd.append([[]])
            for coords_batch in coords[i].decomposed_features:
                scene_min = coords_batch.min(dim=0)[0][None, ...]
                scene_max = coords_batch.max(dim=0)[0][None, ...]

                with autocast(enabled=False):
                    tmp = self.pos_enc(coords_batch[None, ...].float(),
                                       input_range=[scene_min, scene_max])

                pos_encodings_pcd[-1][0].append(tmp.squeeze(0).permute((1, 0)))

        return pos_encodings_pcd
    

    def forward_eval(self,data,raw_coordinates=None,point2segment=None,target=None):
        feature = self.backbone(data)
        #pred_point = self.cls(feature.F)
        torch.cuda.empty_cache()
        with torch.no_grad():
            coordinates = me.SparseTensor(features=torch.cat(raw_coordinates).type(torch.float32),
                                          coordinate_manager=feature.coordinate_manager,
                                          coordinate_map_key=feature.coordinate_map_key,
                                          device=feature.device)
        pos_encodings_pcd = self.get_pos_encs([coordinates])
        mask_segments = []
        mask_seg_coords = []
        torch.cuda.empty_cache()
        for i, mask_feature in enumerate(feature.decomposed_features):
            torch.cuda.empty_cache()
            mask_segments.append(scatter_mean(mask_feature, point2segment[i].type(torch.int64).cuda(), dim=0))
            mask_seg_coords.append(scatter_mean(pos_encodings_pcd[-1][0][i], point2segment[i].type(torch.int64).cuda(), dim=0))
       
        # rand_idx = []
        # mask_idx = []
        preds =  []
        for k in range(len(mask_segments)):
            torch.cuda.empty_cache()
            fps_idx = furthest_point_sample(mask_seg_coords[k][None,...].float(),512).squeeze(0).long()
            key_feat = mask_segments[k][fps_idx]
            key_pos = mask_seg_coords[k][fps_idx]
            query_feat = mask_segments[k][None,...]
            for i in range(2):
                torch.cuda.empty_cache()
                query_feat = self.attention[i](
                    query_feat.transpose(0,1),key_feat[None,...],
                    pos=key_pos[None,...],
                    query_pos=mask_seg_coords[k][None,...].transpose(0,1)
                ).transpose(0,1)
            feat = query_feat[0][point2segment[k].long()]
            weight = self.cls1(torch.abs(feat-feature.decomposed_features[k])).sigmoid()
            preds.append(self.cls((1-weight)*feature.decomposed_features[k] + feat*weight))
        preds = torch.cat(preds)
        torch.cuda.empty_cache()
        return preds
           
    def forward_eval_s3dis(self,data,raw_coordinates=None,point2segment=None,target=None):
        feature = self.backbone(data)
        pred_point = self.cls(feature.F)
        torch.cuda.empty_cache()
        with torch.no_grad():
            coordinates = me.SparseTensor(features=raw_coordinates,
                                          coordinate_manager=feature.coordinate_manager,
                                          coordinate_map_key=feature.coordinate_map_key,
                                          device=feature.device)
        pos_encodings_pcd = self.get_pos_encs([coordinates])
        mask_segments = []
        mask_seg_coords = []
        torch.cuda.empty_cache()
        for i, mask_feature in enumerate(feature.decomposed_features):
            torch.cuda.empty_cache()
            mask_segments.append(scatter_mean(mask_feature.detach().clone(), point2segment[i].type(torch.int64).cuda(), dim=0))
            mask_seg_coords.append(scatter_mean(pos_encodings_pcd[-1][0][i].detach().clone(), point2segment[i].type(torch.int64).cuda(), dim=0))
       
        # rand_idx = []
        # mask_idx = []
        preds =  []
        for k in range(len(mask_segments)):
            torch.cuda.empty_cache()
            fps_idx = furthest_point_sample(mask_seg_coords[k][None,...].float(),128).squeeze(0).long()
            key_feat = mask_segments[k][fps_idx]
            key_pos = mask_seg_coords[k][fps_idx]
            query_feat = mask_segments[k][None,...]
            if query_feat.shape[1]>2000:
                split_num = query_feat.shape[1]//2000+1
                a_list = []
                for j in range(split_num):
                    a = query_feat[:,2000*j:2000*(j+1)]
                    a_pos = mask_seg_coords[k][2000*j:2000*(j+1)]
                    for i in range(2):
                        torch.cuda.empty_cache()
                        a = self.attention[i](
                            a.transpose(0,1),key_feat[None,...],
                            pos=key_pos[None,...],
                            query_pos=a_pos[None,...].transpose(0,1)
                        ).transpose(0,1)
                    a_list.append(a)
                query_feat = torch.cat(a_list,dim=1)
            else:
                for i in range(2):
                    torch.cuda.empty_cache()
                    query_feat = self.attention[i](
                        query_feat.transpose(0,1),key_feat[None,...],
                        pos=key_pos[None,...],
                        query_pos=mask_seg_coords[k][None,...].transpose(0,1)
                    ).transpose(0,1)
            torch.cuda.empty_cache()
            feat = query_feat[0][point2segment[k].long()]# M*C --> N*C
            torch.cuda.empty_cache()
            weight = self.cls1(torch.abs(feat-feature.decomposed_features[k])).sigmoid()
            preds.append(self.cls((1-weight)*feature.decomposed_features[k] + feat*weight))
        preds = torch.cat(preds)
        # weight = self.cls1(torch.abs(off-feature.F)).sigmoid()
        # preds =  self.cls(feature.F + off*weight)
        # loss =  (F.cross_entropy(preds,target.long().cuda(),ignore_index=-100) + F.cross_entropy(pred_point,target.long().cuda(),ignore_index=-100))/2
        # torch.cuda.empty_cache()
        return preds
                   

        
class DMLayer(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.0,
                 activation="relu", normalize_before=False,feat_size=512):
        super().__init__()
        self.dm_attn = DMModule(d_model, nhead, dropout=dropout,match_dim=d_model,feat_size=feat_size)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     memory_mask = None,
                     memory_key_padding_mask = None,
                     pos = None,
                     query_pos = None):
        tgt2 = self.dm_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)

        return tgt

    def forward_pre(self, tgt, memory,
                    memory_mask = None,
                    memory_key_padding_mask = None,
                    pos = None,
                    query_pos = None):
        tgt2 = self.norm(tgt)

        tgt2 = self.dm_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask,pos_emb=query_pos)[0]
        tgt = tgt + self.dropout(tgt2)

        return tgt

    def forward(self, tgt, memory,
                memory_mask = None,
                memory_key_padding_mask = None,
                pos = None,
                query_pos = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, memory_mask,
                                    memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, memory_mask,
                                 memory_key_padding_mask, pos, query_pos)
def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")