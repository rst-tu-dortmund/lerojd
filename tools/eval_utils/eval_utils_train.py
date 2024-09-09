import pickle
import time

import numpy as np
import torch
import tqdm

from pcdet.models import load_data_to_gpu
from pcdet.utils import common_utils
from pcdet.utils.kd_utils import kd_forwad

try:
    from thop import clever_format
except:
    pass
    # you cannot use cal_param without profile


def statistics_info(RECALL_THRESH_LIST, ret_dict, metric, disp_dict):
    for cur_thresh in RECALL_THRESH_LIST:
        metric['recall_roi_%s' % str(cur_thresh)] += ret_dict.get('roi_%s' % str(cur_thresh), 0)
        metric['recall_rcnn_%s' % str(cur_thresh)] += ret_dict.get('rcnn_%s' % str(cur_thresh), 0)
    metric['gt_num'] += ret_dict.get('gt', 0)
    min_thresh = RECALL_THRESH_LIST[0]
    disp_dict['recall_%s' % str(min_thresh)] = \
        '(%d, %d) / %d' % (metric['recall_roi_%s' % str(min_thresh)], metric['recall_rcnn_%s' % str(min_thresh)], metric['gt_num'])


def eval_one_epoch_train(model, dataloader):
    RECALL_THRESH_LIST = [0.3, 0.5, 0.7]
    LOCAL_RANK = 0
    EVAL_METRIC = 'VoD'

    metric = {
        'gt_num': 0,
    }
    # for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
    #     metric['recall_roi_%s' % str(cur_thresh)] = 0
    #     metric['recall_rcnn_%s' % str(cur_thresh)] = 0

    dataset = dataloader.dataset
    class_names = dataset.class_names
    det_annos = []

    model.eval()
    # model.apply(common_utils.set_bn_train)

    start_time = time.time()
    # for i, batch_dict in enumerate(dataloader):
    #     load_data_to_gpu(batch_dict)
    #     with torch.no_grad():
    #         pred_dicts, ret_dict = model(batch_dict, record_time=getattr(args, 'infer_time', False) and i > start_iter)
    #     disp_dict = {}

    for i, batch_dict in enumerate(dataloader):
        load_data_to_gpu(batch_dict)
        with torch.no_grad():
            pred_dicts, ret_dict = model(batch_dict)
        disp_dict = {}

        # statistics_info(cfg, ret_dict, metric, disp_dict)
        annos = dataset.generate_prediction_dicts(
            batch_dict, pred_dicts, class_names,
            output_path=None
        )
        det_annos += annos


    # ret_dict = {}


    # gt_num_cnt = metric['gt_num']
    # for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
    #     cur_roi_recall = metric['recall_roi_%s' % str(cur_thresh)] / max(gt_num_cnt, 1)
    #     cur_rcnn_recall = metric['recall_rcnn_%s' % str(cur_thresh)] / max(gt_num_cnt, 1)
    #     logger.info('recall_roi_%s: %f' % (cur_thresh, cur_roi_recall))
    #     logger.info('recall_rcnn_%s: %f' % (cur_thresh, cur_rcnn_recall))
    #     ret_dict['recall/roi_%s' % str(cur_thresh)] = cur_roi_recall
    #     ret_dict['recall/rcnn_%s' % str(cur_thresh)] = cur_rcnn_recall

    total_pred_objects = 0
    for anno in det_annos:
        total_pred_objects += anno['name'].__len__()


    # result_str, result_dict = dataset.evaluation(
    #     det_annos, class_names,
    #     eval_metric=cfg.MODEL.POST_PROCESSING.EVAL_METRIC,
    #     output_path=final_output_dir
    # )

    # result_dict = dataset.evaluation(
    #     det_annos, class_names
    # )
    result_str, result_dict = dataset.evaluation(
        det_annos, class_names
    )

    ret_dict.update(result_dict)

    return ret_dict


def get_multi_classes_mAP(result_dict, result_str, metric_dict):
    result_str += '\nmAP\n'
    for metric, class_list in metric_dict.items():
        mAP = 0
        for cls in class_list:
            mAP += result_dict[cls]
        mAP /= len(class_list)
        result_dict['mAP/' + metric] = mAP
        result_str += metric + ' mAP: {:.4f}\n'.format(mAP)

    return result_dict, result_str


if __name__ == '__main__':
    pass
