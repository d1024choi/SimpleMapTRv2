import sys
import torch
import torch.nn.functional as F
from torchvision import models
from torch import nn
from models.common import xavier_init
import copy    
from models.common import inverse_sigmoid
from models.common import bbox_xyxy_to_cxcywh, normalize_2d_bbox, denormalize_2d_bbox, normalize_2d_pts, denormalize_2d_pts
from models.common import bias_init_with_prob, reduce_mean

# NOTE : This part should correspond to class MapTRHead() (/project/mmdet3d_plugin/maptr/dense_heads/maptr_head.py)
class MapTR(nn.Module):
    def __init__(self, cfg, model_name='Scratch', dataset_name='nuscenes', isTrain=False):
        super().__init__()

        # cfg initialization
        self.isTrain = isTrain
        self.map_decoder_type = 'maptr'
        cfg_model = cfg[model_name]
        cfg_ds = cfg[dataset_name]
        cfg_bev = cfg_ds['bev']       
        cfg_map = cfg['MapDecoder'][self.map_decoder_type]     
        self.cfg_one2many = cfg_model["aux_tasks"]["one2many"]   
        
        # Parameters
        self.bev_h = cfg_bev['bev_h']
        self.bev_w = cfg_bev['bev_w']
        # self.num_vec_one2many = cfg_one2many['num_vec_one2many'] # "one2many": {"flag": true, "k_one2many": 6, "num_vec_one2many": 300}
        self.h_dim = cfg_model['h_dim']
        self.num_layers = cfg_map['num_layers']
        self.query_embed_type = cfg_map['query_embed_type']
        self.cls_out_channels = cfg_ds['OnlineHDmap']['polyline']['num_classes']
        self.code_size = cfg_ds['OnlineHDmap']['polyline']['code_size']
        self.num_reg_fcs = cfg_map['num_reg_fcs']
        self.with_box_refine = cfg_map['with_box_refine']
        self.transform_method = cfg_map['transform_method']
        self.num_classes = cfg_ds['OnlineHDmap']['polyline']['num_classes']
        self.pc_range = cfg_ds['pc_range']
        self.mixed_precision = cfg['train']['bool_mixed_precision']
        cfg_map.update({"mixed_precision": self.mixed_precision})

        self.num_vec_one2one = cfg_map['num_vec'] # 50
        num_vec = self.num_vec_one2one
        if (self.cfg_one2many['flag']):
            num_vec += self.cfg_one2many['num_vec_one2many'] # 50 + 300
        self.num_vec = num_vec
        self.num_pts_per_vec = cfg_ds['OnlineHDmap']['polyline']['fixed_ptsnum_per_line'] # 20
        self.num_query = self.num_pts_per_vec * num_vec # 20 * (50 + 300)
        


        # Model instantiation
        from models.map_decoder.maptr.maptr_decoder import MapTRDecoder
        cfg_map.update({"num_vec": num_vec, "num_pts_per_vec": self.num_pts_per_vec, "num_vec_one2one": self.num_vec_one2one, "isTrain": isTrain})
        self.decoder = MapTRDecoder(**cfg_map)
        self.reference_points = nn.Linear(self.h_dim, 2)


        # # +++++++++++++++++++++++++++++++++++++
        # # 260327
        # # +++++++++++++++++++++++++++++++++++++

        # LossComputation (following maptr config)
        if isTrain:
            from utils.loss import Assigner
            # Assigner
            cfg_assign = {
                'cls_weight': 2.0,
                'use_sigmoid': True,
                'line_weight': 5.0,
                'line_cost_type': 'l1',
                'mask_weight': 1.0,
                'mask_chunk_size': 16,
                'pc_range': cfg_ds['pc_range'],
                'bbox_weight': 0.0,
                'num_classes': self.num_classes
            }
            self.assigner = Assigner(cfg_assign)

            # LossComputation (following mask2map config)
            cfg_loss = {
                'loss_cls': {
                    'use_sigmoid': True,
                    'gamma': 2.0,
                    'alpha': 0.25,
                    'loss_weight': 2.0,
                    'bg_cls_weight': 0.1,
                    'target' : cfg_ds['OnlineHDmap']['target_classes'] # ["divider","ped_crossing","boundary"]
                },
                'loss_pts': {'loss_weight': 5.0},
                'loss_dir': {'loss_weight': 0.005},
                'loss_mask': {
                    'loss_weight': 1.0,
                    'alpha': 0.25,
                    'gamma': 2.0
                },
                'dir_interval': 1,
                'sync_cls_avg_factor': True
            }
            from utils.loss import LossComputation
            self.loss_calc = LossComputation(cfg_loss)            


        self._init_layers()
        self._init_weights()


    def _init_layers(self):

        """Initialize classification branch and regression branch of head."""

        if self.query_embed_type == 'all_pts':
            self.query_embedding = nn.Embedding(self.num_query, self.h_dim * 2)
        elif self.query_embed_type == 'instance_pts':
            self.query_embedding = None
            self.instance_embedding = nn.Embedding(self.num_vec, self.h_dim * 2)
            self.pts_embedding = nn.Embedding(self.num_pts_per_vec, self.h_dim * 2)

        cls_branch = []
        for _ in range(self.num_reg_fcs):
            cls_branch.append(nn.Linear(self.h_dim, self.h_dim))
            cls_branch.append(nn.LayerNorm(self.h_dim))
            cls_branch.append(nn.ReLU(inplace=True))
        cls_branch.append(nn.Linear(self.h_dim, self.cls_out_channels))
        fc_cls = nn.Sequential(*cls_branch)

        reg_branch = []
        for _ in range(self.num_reg_fcs):
            reg_branch.append(nn.Linear(self.h_dim, self.h_dim))
            reg_branch.append(nn.ReLU())
        reg_branch.append(nn.Linear(self.h_dim, self.code_size))
        reg_branch = nn.Sequential(*reg_branch)

        def _get_clones(module, N):
            return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

        # last reg_branch is used to generate proposal from
        # encode feature map when as_two_stage is True.
        self.as_two_stage = False
        num_pred = (self.num_layers + 1) if self.as_two_stage else self.num_layers

        if self.with_box_refine:
            self.cls_branches = _get_clones(fc_cls, num_pred)
            self.reg_branches = _get_clones(reg_branch, num_pred)
        else:
            self.cls_branches = nn.ModuleList(
                [fc_cls for _ in range(num_pred)])
            self.reg_branches = nn.ModuleList(
                [reg_branch for _ in range(num_pred)])

    def _init_weights(self):
        """Initialize the transformer weights."""
        xavier_init(self.reference_points, distribution='uniform', bias=0.)

        # NOTE : Initialize the classification branch bias (260415)
        bias_init = bias_init_with_prob(0.01)
        for m in self.cls_branches:
            nn.init.constant_(m[-1].bias, bias_init)

    def decoding(self, bev_embed, query, query_pos):
        '''
        bev_embed : b (bev_h*bev_w) dim
        query_pos : bs, num_query, h_dim
        query : bs, num_query, h_dim
        '''

        bs = query.size(0)

        reference_points = self.reference_points(query_pos) # (bs, num_query, 2)
        reference_points = reference_points.sigmoid() # 
        init_reference= reference_points

        # from (bs, num_query, h_dim) to (num_query, bs, h_dim)
        query = query.permute(1, 0, 2)
        query_pos = query_pos.permute(1, 0, 2)
        bev_embed = bev_embed.permute(1, 0, 2)

        # NOTE : class MapTRHead()
        # hs : num_layers num_total_queries bs h_dim
        hs, inter_references = self.decoder(query=query, 
                                            key=None, 
                                            value=bev_embed, 
                                            query_pos=query_pos, 
                                            reference_points=reference_points, # sigmoid space
                                            reg_branches=self.reg_branches,
                                            spatial_shapes=torch.tensor([[self.bev_h, self.bev_w]], device=query.device), 
                                            level_start_index=torch.tensor([0], device=query.device))
        hs = hs.permute(0, 2, 1, 3) # hs : num_layers bs num_total_queries h_dim                                            

        # NOTE : MapTRHead (../dense_heads/maptr_head.py)
        outputs_classes, outputs_coords, outputs_pts_coords = [], [], []
        for lvl in range(hs.shape[0]):

            # get reference points
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference) # in Logit space

            # get class
            outputs_class = self.cls_branches[lvl](hs[lvl].view(bs, self.num_vec, self.num_pts_per_vec, -1).mean(2))

            # get coordinates
            tmp = self.reg_branches[lvl](hs[lvl])
            tmp[..., 0:2] += reference[..., 0:2] # in Logit space
            tmp = tmp.sigmoid() 

            # TODO : transform box and normalization functions should be updated!!!!
            outputs_coord, outputs_pts_coord = self.transform_box(tmp)
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
            outputs_pts_coords.append(outputs_pts_coord)

        outputs_classes = torch.stack(outputs_classes) 
        outputs_coords = torch.stack(outputs_coords) # (cx, cy, W, H)
        outputs_pts_coords = torch.stack(outputs_pts_coords) # (x, y), the width and height in BEV space
        
        outs = {'all_cls_scores': outputs_classes,
                'all_bbox_preds': outputs_coords,
                'all_pts_preds': outputs_pts_coords}

        return outs

    def forward(self, bev_embed, isTrain=False):
        '''
        bev_embed : b (bev_h*bev_w) dim

          (Top-down view grid for bev_embed) 

         -15         15
      -30 -------------
          |     |     |
          |     |     |
          -------------> x (100 pels)
          |     |     |   
          |     |     |            
      +30 ------v------
                y
             (200 pels)                
        
        ** How pipeline goes in the original implementation **
        (1) class MapTR() in ...maptr/detectors/maptr.py
        (2) class MapTRHead() in ...maptr/dense_heads/maptr_head.py
        (3) class MapTRPerceptaionTransformer() in ...maptr/modules/transformer.py
        (4) class MapTRDecoder() in ...maptr/modules/decoder.py
        
        '''

        dtype = bev_embed.dtype
        bs = bev_embed.size(0)
        
        if self.isTrain:
            num_vec = self.num_vec
        else:
            num_vec = self.num_vec_one2one

        # NOTE : class MapTRHead()
        if self.query_embed_type == 'all_pts':
            object_query_embeds = self.query_embedding.weight.to(dtype)
        elif self.query_embed_type == 'instance_pts':
            pts_embeds = self.pts_embedding.weight.unsqueeze(0)
            instance_embeds = self.instance_embedding.weight[0:num_vec].unsqueeze(1)
            object_query_embeds = (pts_embeds + instance_embeds).flatten(0, 1).to(dtype) # (num_vec * num_pts_per_vec, 2*h_dim)


        # NOTE : class MapTRPerceptaionTransformer()
        query_pos, query = torch.split(object_query_embeds, self.h_dim, dim=1) # (num_query, h_dim) & (num_query, h_dim)
        query_pos = query_pos.unsqueeze(0).expand(bs, -1, -1)
        query = query.unsqueeze(0).expand(bs, -1, -1)

        reference_points = self.reference_points(query_pos) # (bs, num_query, 2)
        reference_points = reference_points.sigmoid() # 
        init_reference= reference_points

        # from (bs, num_query, h_dim) to (num_query, bs, h_dim)
        query = query.permute(1, 0, 2)
        query_pos = query_pos.permute(1, 0, 2)
        bev_embed = bev_embed.permute(1, 0, 2)

        # make attn mask
        """ attention mask to prevent information leakage """
        # NOTE: keep dtype=bool. Casting to bev_embed's float dtype turns this
        # into an *additive* mask in nn.MultiheadAttention (blocked positions
        # get +1.0 added instead of -inf), which silently defeats the block
        # between one2one and one2many queries.
        self_attn_mask = torch.zeros(
            (num_vec, num_vec), dtype=torch.bool, device=bev_embed.device
        )
        self_attn_mask[self.num_vec_one2one :, 0 : self.num_vec_one2one,] = True
        self_attn_mask[0 : self.num_vec_one2one, self.num_vec_one2one :,] = True
        # self_attn_mask = None

        # NOTE : class MapTRHead()
        # hs : num_layers num_total_queries bs h_dim
        hs, inter_references = self.decoder(query=query, 
                                            key=None, 
                                            value=bev_embed, 
                                            query_pos=query_pos, 
                                            reference_points=reference_points, # sigmoid space
                                            reg_branches=self.reg_branches,
                                            attn_masks=self_attn_mask,
                                            spatial_shapes=torch.tensor([[self.bev_h, self.bev_w]], device=query.device), 
                                            level_start_index=torch.tensor([0], device=query.device))
        hs = hs.permute(0, 2, 1, 3) # hs : num_layers bs num_total_queries h_dim                                            

        # NOTE : MapTRHead (../dense_heads/maptr_head.py)
        outputs_classes, outputs_coords, outputs_pts_coords = [], [], []
        for lvl in range(hs.shape[0]):

            # get reference points
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference) # in Logit space

            # get class
            outputs_class = self.cls_branches[lvl](hs[lvl].view(bs, num_vec, self.num_pts_per_vec, -1).mean(2))

            # get coordinates
            tmp = self.reg_branches[lvl](hs[lvl])
            tmp[..., 0:2] += reference[..., 0:2] # in Logit space
            tmp = tmp.sigmoid() 

            # TODO : transform box and normalization functions should be updated!!!!
            outputs_coord, outputs_pts_coord = self.transform_box(tmp)
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
            outputs_pts_coords.append(outputs_pts_coord)

        outputs_classes = torch.stack(outputs_classes) 
        outputs_coords = torch.stack(outputs_coords) # (cx, cy, W, H)
        outputs_pts_coords = torch.stack(outputs_pts_coords) # (x, y), the width and height in BEV space
        
        outs_one2one = {'all_cls_scores': outputs_classes[:, :, :self.num_vec_one2one],
                        'all_bbox_preds': outputs_coords[:, :, :self.num_vec_one2one],
                        'all_pts_preds': outputs_pts_coords[:, :, :self.num_vec_one2one],
                        'k_one2many': 1}

        outs_one2many = None
        if (self.cfg_one2many['flag'] and isTrain):
            outs_one2many = {'all_cls_scores': outputs_classes[:, :, self.num_vec_one2one:],
                             'all_bbox_preds': outputs_coords[:, :, self.num_vec_one2one:],
                             'all_pts_preds': outputs_pts_coords[:, :, self.num_vec_one2one:],
                             'k_one2many': self.cfg_one2many['k_one2many']}


        return {'outs_one2one': outs_one2one, 'outs_one2many': outs_one2many}



    def transform_box(self, pts, y_first=False):
        """
        Converting the points set into bounding box.

        Args:
            pts: the input points sets (fields), each points
                set (fields) is represented as 2n scalar.
            y_first: if y_fisrt=True, the point set is represented as
                [y1, x1, y2, x2 ... yn, xn], otherwise the point set is
                represented as [x1, y1, x2, y2 ... xn, yn].
        Returns:
            The bbox [cx, cy, w, h] transformed from points.
        """
        pts_reshape = pts.view(pts.shape[0], -1, self.num_pts_per_vec, 2)
        pts_y = pts_reshape[:, :, :, 0] if y_first else pts_reshape[:, :, :, 1]
        pts_x = pts_reshape[:, :, :, 1] if y_first else pts_reshape[:, :, :, 0]
        if self.transform_method == 'minmax':
            xmin = pts_x.min(dim=2, keepdim=True)[0]
            xmax = pts_x.max(dim=2, keepdim=True)[0]
            ymin = pts_y.min(dim=2, keepdim=True)[0]
            ymax = pts_y.max(dim=2, keepdim=True)[0]
            bbox = torch.cat([xmin, ymin, xmax, ymax], dim=2)
            bbox = bbox_xyxy_to_cxcywh(bbox)
        else:
            raise NotImplementedError
        return bbox, pts_reshape


    def loss(self, preds_dicts, gt_label, batch_size):
        """Loss function.

        ** class MapTRHead() in ...maptr/dense_heads/maptr_head.py **
        """

        final_loss = {'lane_loss': 0.0, 'loss_cls': 0.0, 'loss_pts': 0.0, 'loss_dir': 0.0, 'loss_mask': 0.0}
        
        k_one2many = preds_dicts['k_one2many']
        valid_samples_in_batch = []
        gt_labels_list, gt_bboxes_list, gt_pts_list, gt_shifts_pts_list = [], [], [], []
        for b in range(batch_size):
            _, gt_dict = max(gt_label[b].items())
            
            if (len(gt_dict['bboxes']) == 0):
                valid_samples_in_batch.append(False)
            else:
                valid_samples_in_batch.append(True)
            # each element corresponds to one sample in a batch
            try:
                gt_bboxes_list.append(gt_dict['bboxes'].repeat(k_one2many, 1)) 
                gt_pts_list.append(gt_dict['polylines'].repeat(k_one2many, 1, 1))
                gt_shifts_pts_list.append(gt_dict['polylines_shift'].repeat(k_one2many, 1, 1, 1))
                gt_labels_list.append(gt_dict['labels'].repeat(k_one2many))
            except:
                gt_bboxes_list.append(gt_dict['bboxes']) 
                gt_pts_list.append(gt_dict['polylines'])
                gt_shifts_pts_list.append(gt_dict['polylines_shift'])
                gt_labels_list.append(gt_dict['labels'])

        # When the ground-truth lists are empty.
        if (all(not valid for valid in valid_samples_in_batch)):
            dummy_loss = 0.0 * (all_cls_scores.sum() + all_bbox_preds.sum() + all_pts_preds.sum())
            for l in range(self.num_layers):
                for key in final_loss.keys():
                    final_loss[key] += dummy_loss             
                final_loss[f'{l}_layer_'+key] = dummy_loss
            
            return final_loss


        all_cls_scores = preds_dicts['all_cls_scores'] # num_dec_layers, bs, num_total_queries, num_classes
        all_bbox_preds = preds_dicts['all_bbox_preds'] # num_dec_layers, bs, num_total_queries, 4
        all_pts_preds  = preds_dicts['all_pts_preds']  # num_dec_layers, bs, num_total_queries, num_pts_per_vec, 2
        

        # NOTE : iterate over all decoding layers
        for l in range(self.num_layers):


            num_total_pos, num_total_neg = 0, 0
            pos_list, cls_list, bbox_list, mask_list = [], [], [], []
            labels_list, label_weights_list, pos_targets_list, pos_weights_list, bbox_targets_list, bbox_weights_list, mask_targets_list, mask_weights_list \
                = [], [], [], [], [], [], [], []

            # NOTE : iterate over all samples in a batch
            single_layer = self.single_layer(all_cls_scores[l], all_bbox_preds[l], all_pts_preds[l], gt_bboxes_list, 
                                             gt_pts_list, gt_shifts_pts_list, gt_labels_list, valid_samples_in_batch)

            pos_list.append(single_layer['pos'])
            cls_list.append(single_layer['cls'])
            bbox_list.append(single_layer['bbox'])
            labels_list.append(single_layer['labels_list'])
            label_weights_list.append(single_layer['label_weights'])
            pos_targets_list.append(single_layer['pos_targets'])
            pos_weights_list.append(single_layer['pos_weights'])
            bbox_targets_list.append(single_layer['bbox_targets'])
            bbox_weights_list.append(single_layer['bbox_weights'])

            num_total_pos += single_layer['num_total_pos']
            num_total_neg += single_layer['num_total_neg']
            
            # Stack predictions and targets for the whole layers
            if (num_total_pos > 0):
                preds = {'pos': torch.cat(pos_list, dim=0), 
                        'cls': torch.cat(cls_list, dim=0)
                }
                targets = {
                    'labels': torch.cat(labels_list, dim=0),
                    'label_weights': torch.cat(label_weights_list, dim=0),
                    'pos_targets': torch.cat(pos_targets_list, dim=0),
                    'pos_weights': torch.cat(pos_weights_list, dim=0)
                }

                # Calculate the loss for the current batch
                layer_loss = self.loss_calc(preds, targets, num_total_pos, num_total_neg)
                for key in layer_loss.keys():
                    final_loss[key] += layer_loss[key]
                    final_loss[f'{l}_layer_'+key] = layer_loss[key]
            else:
                dummy_loss = 0.0 * (all_cls_scores.sum() + all_bbox_preds.sum() + all_pts_preds.sum())
                for key in final_loss.keys():
                    final_loss[key] += dummy_loss    


        return final_loss        

    def single_layer(self, cls_scores, bbox_preds, pts_preds, gt_bboxes_list, gt_pts_list, gt_shifts_pts_list, gt_labels_list, valid_samples_in_batch):
        ''' Consider only a single layer

        cls_scores : bs num_total_queries num_classes
        bbox_preds : bs num_total_queries 4, ** normalzed (xyhw) format
        pts_preds : bs num_total_queries num_pts_per_vec 2, ** normalized (xy) format
        gt_bboxes_list : list of (num_gts 4), ** unnormalized (xyxy) format
        gt_pts_list : list of (num_gts num_pts_per_vec 2), ** unnormalized (xy) format
        gt_shifts_pts_list : list of (num_gts num_shifts num_pts_per_vec 2), ** unnormalized (xy) format
        gt_labels_list : list of (num_gts)
        '''

        num_total_pos, num_total_neg = 0, 0
        pos_list, cls_list, bbox_list, mask_list = [], [], [], []
        labels_list, label_weights_list, pos_targets_list, pos_weights_list, bbox_targets_list, bbox_weights_list, mask_targets_list, mask_weights_list = [], [], [], [], [], [], [], []

        bs = cls_scores.size(0)
        for b in range(bs):
            if not valid_samples_in_batch[b]:
                continue

            cls_score = cls_scores[b]
            bbox_pred = bbox_preds[b]
            pts_pred = pts_preds[b]
            gt_bboxes = gt_bboxes_list[b]
            gt_pts = gt_pts_list[b]
            gt_shifts_pts = gt_shifts_pts_list[b]
            gt_labels = gt_labels_list[b]
            
            target_single = self._get_target_single(pts_pred, cls_score, bbox_pred, gt_shifts_pts, gt_labels, gt_bboxes)

            pos_list.append(pts_pred) # num_queries num_points 2
            cls_list.append(cls_score) # num_queries num_classes
            bbox_list.append(bbox_pred) # num_queries 4
            # mask_list.append(None) - Skip for now
            labels_list.append(target_single['labels']) # num_queries
            label_weights_list.append(target_single['label_weights']) # num_queries
            pos_targets_list.append(target_single['pos_targets']) # num_queries num_points 2
            pos_weights_list.append(target_single['pos_weights']) # num_queries num_points 2
            bbox_targets_list.append(target_single['bbox_targets']) # num_queries 4
            bbox_weights_list.append(target_single['bbox_weights']) # num_queries 4
            # mask_targets_list.append(target_single['mask_targets']) # Skip for now
            # mask_weights_list.append(target_single['mask_weights']) # Skip for now
            num_total_pos += target_single['pos_inds'].numel()
            num_total_neg += target_single['neg_inds'].numel()


        output = {
            'pos': torch.stack(pos_list, dim=0),
            'pos_targets': torch.stack(pos_targets_list, dim=0),
            'pos_weights': torch.stack(pos_weights_list, dim=0),

            'cls': torch.stack(cls_list, dim=0),
            'labels_list': torch.stack(labels_list, dim=0),
            'label_weights': torch.stack(label_weights_list, dim=0),

            'bbox': torch.stack(bbox_list, dim=0),
            'bbox_targets': torch.stack(bbox_targets_list, dim=0),
            'bbox_weights': torch.stack(bbox_weights_list, dim=0),

            'num_total_pos': num_total_pos,
            'num_total_neg': num_total_neg
        }
        return output

    def _get_target_single(self, pred_pos, pred_cls, pred_bbox, gt_pos_shift, gt_cls, gt_bbox):
        '''
        ** Consider only a sample data in a batch **

        pred_pos : num_queries num_points 2, ** normalized (xy) format
        pred_cls : num_queries num_classes 
        pred_bbox : num_queries 4, ** normalzed (xyhw) format
        gt_pos_shift_norm : num_gt num_shifts num_points 2, ** unnormalized (xy) format
        gt_cls : num_gt (index does not include background class) **
        gt_bbox : num_gt 4, ** unnormalized (xyxy) format
        '''


        num_gts, pos_inds, neg_inds, pos_assigned_gt_inds, order_index = self.assigner.assign(pred_pos, pred_cls, pred_bbox, gt_pos_shift, gt_cls, gt_bbox)

        gt_cls_gpu = gt_cls.to(pred_pos.device)
        gt_pos_shift_norm_gpu = normalize_2d_pts(gt_pos_shift, self.pc_range).to(pred_pos.device) # debug, (260320)
        gt_bbox_norm_gpu = normalize_2d_bbox(gt_bbox, self.pc_range).to(pred_pos.device) # debug, (260320)

        # labels : num_queries, label_weights : num_queries
        num_queries = pred_pos.size(0)
        labels = pred_pos.new_full((num_queries,), self.num_classes, dtype=torch.long)
        labels[pos_inds] = gt_cls_gpu[pos_assigned_gt_inds].long()
        label_weights = pred_pos.new_ones(num_queries)

        # pts targets (cast GT to pred dtype for index_put under mixed precision)
        assigned_shift = order_index[pos_inds, pos_assigned_gt_inds].long()
        pos_targets = pred_pos.new_zeros((pred_pos.size(0), pred_pos.size(1), pred_pos.size(2)))
        pos_weights = torch.zeros_like(pos_targets)
        pos_weights[pos_inds] = 1.0
        pos_targets[pos_inds] = gt_pos_shift_norm_gpu[pos_assigned_gt_inds, assigned_shift, :, :].to(pos_targets.dtype)

        # bbox targets (cast GT to pred dtype for index_put under mixed precision)
        bbox_targets = pred_bbox.new_zeros((pred_bbox.size(0), pred_bbox.size(1)))
        bbox_weights = torch.zeros_like(bbox_targets)
        bbox_weights[pos_inds] = 1.0
        bbox_targets[pos_inds] = gt_bbox_norm_gpu[pos_assigned_gt_inds].to(bbox_targets.dtype)



        # # mask targets - Deprecated
        # mask_targets = pred_mask.new_zeros((pred_mask.size(0), pred_mask.size(1), pred_mask.size(2)))
        # mask_weights = torch.zeros_like(mask_targets)
        # mask_weights[pos_inds] = 1.0            
        # mask_targets[pos_inds] = gt_mask[pos_assigned_gt_inds]
        mask_targets = None
        mask_weights = None

        return {
            'labels': labels,
            'label_weights': label_weights,
            'pos_targets': pos_targets,
            'pos_weights': pos_weights,
            'bbox_targets': bbox_targets,
            'bbox_weights': bbox_weights,
            'mask_targets': mask_targets,
            'mask_weights': mask_weights,
            'pos_inds': pos_inds,
            'neg_inds': neg_inds
        }

