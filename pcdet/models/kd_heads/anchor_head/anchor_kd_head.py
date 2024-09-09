import torch
import torch.nn as nn
import numpy as np

from .anchor_logit_kd_head import AnchorLogitKDHead
from .anchor_feature_kd_head import AnchorFeatureKDHead
from .anchor_label_kd_head import AnchorLabelAssignKDHead

from ...model_utils import model_nms_utils



class AnchorHeadKD(AnchorLogitKDHead, AnchorFeatureKDHead, AnchorLabelAssignKDHead):
    def __init__(self, model_cfg, dense_head):
        super(AnchorHeadKD, self).__init__(model_cfg, dense_head)
        self.build_loss(dense_head)

    def get_kd_loss(self, batch_dict, tb_dict):
        kd_loss = 0.0
        if self.model_cfg.get('LOGIT_KD', None) and self.model_cfg.LOGIT_KD.ENABLED:
            kd_logit_loss, tb_dict = self.get_logit_kd_loss(batch_dict, tb_dict)
            kd_loss += kd_logit_loss

        if self.model_cfg.get('FEATURE_KD', None) and self.model_cfg.FEATURE_KD.ENABLED:
            kd_feature_loss, tb_dict = self.get_feature_kd_loss(
                batch_dict, tb_dict, self.model_cfg.KD_LOSS.FEATURE_LOSS
            )
            kd_loss += kd_feature_loss

        return kd_loss, tb_dict

    def put_pred_to_ret_dict(self, dense_head, data_dict, cls_preds, box_preds):

        if data_dict.get('teacher_decoded_pred_flag', None) and dense_head.training:
            decoded_pred_cls, decoded_pred_box = dense_head.generate_predicted_boxes(
                data_dict['batch_size'], cls_preds, box_preds
            )

            # match dictionary
            # ret_dict = [{
            #     'pred_boxes': [],
            #     'pred_scores': [],
            #     'pred_labels': [],
            # } for k in range(batch_size)]
            #
            # for batch in range(decoded_pred_cls):
            #     for k in range(decoded_pred_cls[batch]):
            #         # ret_dict['pred_labels']
            #
            #         final_dict['pred_boxes'] = final_dict['pred_boxes'][selected]
            #         final_dict['pred_scores'] = selected_scores
            #         final_dict['pred_labels'] = final_dict['pred_labels'][selected]

            data_dict['batch_box_preds'] = decoded_pred_box
            data_dict['batch_cls_preds'] = decoded_pred_cls
            pred_dicts = self.post_processing(data_dict)

            dense_head.forward_ret_dict['decoded_pred_dicts'] = pred_dicts

    def post_processing(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size:
                batch_cls_preds: (B, num_boxes, num_classes | 1) or (N1+N2+..., num_classes | 1)
                                or [(B, num_boxes, num_class1), (B, num_boxes, num_class2) ...]
                multihead_label_mapping: [(num_class1), (num_class2), ...]
                batch_box_preds: (B, num_boxes, 7+C) or (N1+N2+..., 7+C)
                cls_preds_normalized: indicate whether batch_cls_preds is normalized
                batch_index: optional (N1+N2+...)
                has_class_labels: True/False
                roi_labels: (B, num_rois)  1 .. num_classes
                batch_pred_labels: (B, num_boxes, 1)
        Returns:

        """
        post_process_cfg = self.model_cfg.POST_PROCESSING
        batch_size = batch_dict['batch_size']
        # recall_dict = {}
        pred_dicts = []
        num_class = 3
        for index in range(batch_size):
            if batch_dict.get('batch_index', None) is not None:
                # assert batch_dict['batch_box_preds'].shape.__len__() == 2
                batch_mask = (batch_dict['batch_index'] == index)
            else:
                # assert batch_dict['batch_box_preds'].shape.__len__() == 3
                batch_mask = index

            box_preds = batch_dict['batch_box_preds'][batch_mask]
            src_box_preds = box_preds

            if not isinstance(batch_dict['batch_cls_preds'], list):
                cls_preds = batch_dict['batch_cls_preds'][batch_mask]

                src_cls_preds = cls_preds
                assert cls_preds.shape[1] in [1, num_class]

                # if not batch_dict['cls_preds_normalized']:
                #     cls_preds = torch.sigmoid(cls_preds)
                cls_preds = torch.sigmoid(cls_preds)

            else:
                cls_preds = [x[batch_mask] for x in batch_dict['batch_cls_preds']]
                src_cls_preds = cls_preds
                if not batch_dict['cls_preds_normalized']:
                    cls_preds = [torch.sigmoid(x) for x in cls_preds]
                # cls_preds = [torch.sigmoid(x) for x in cls_preds]

            if post_process_cfg.NMS_CONFIG.MULTI_CLASSES_NMS:
                if not isinstance(cls_preds, list):
                    cls_preds = [cls_preds]
                    multihead_label_mapping = [torch.arange(1, num_class, device=cls_preds[0].device)]
                else:
                    multihead_label_mapping = batch_dict['multihead_label_mapping']

                cur_start_idx = 0
                pred_scores, pred_labels, pred_boxes = [], [], []
                for cur_cls_preds, cur_label_mapping in zip(cls_preds, multihead_label_mapping):
                    assert cur_cls_preds.shape[1] == len(cur_label_mapping)
                    cur_box_preds = box_preds[cur_start_idx: cur_start_idx + cur_cls_preds.shape[0]]
                    cur_pred_scores, cur_pred_labels, cur_pred_boxes = model_nms_utils.multi_classes_nms(
                        cls_scores=cur_cls_preds, box_preds=cur_box_preds,
                        nms_config=post_process_cfg.NMS_CONFIG,
                        score_thresh=post_process_cfg.SCORE_THRESH
                    )
                    cur_pred_labels = cur_label_mapping[cur_pred_labels]
                    pred_scores.append(cur_pred_scores)
                    pred_labels.append(cur_pred_labels)
                    pred_boxes.append(cur_pred_boxes)
                    cur_start_idx += cur_cls_preds.shape[0]

                final_scores = torch.cat(pred_scores, dim=0)
                final_labels = torch.cat(pred_labels, dim=0)
                final_boxes = torch.cat(pred_boxes, dim=0)
            else:
                cls_preds, label_preds = torch.max(cls_preds, dim=-1)
                if batch_dict.get('has_class_labels', False):
                    label_key = 'roi_labels' if 'roi_labels' in batch_dict else 'batch_pred_labels'
                    label_preds = batch_dict[label_key][index]
                else:
                    label_preds = label_preds + 1
                selected, selected_scores = model_nms_utils.class_agnostic_nms(
                    box_scores=cls_preds, box_preds=box_preds,
                    nms_config=post_process_cfg.NMS_CONFIG,
                    score_thresh=post_process_cfg.SCORE_THRESH
                )

                if post_process_cfg.OUTPUT_RAW_SCORE:
                    max_cls_preds, _ = torch.max(src_cls_preds, dim=-1)
                    selected_scores = max_cls_preds[selected]

                final_scores = selected_scores
                final_labels = label_preds[selected]
                final_boxes = box_preds[selected]

            record_dict = {
                'pred_boxes': final_boxes,
                'pred_scores': final_scores,
                'pred_labels': final_labels
            }
            pred_dicts.append(record_dict)

        # return pred_dicts, recall_dict
        return pred_dicts

