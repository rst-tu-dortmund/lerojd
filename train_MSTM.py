import os
import shutil
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--folder', default='VoD_MSTM', help='specify model folder (e.g.: VoD_models_1337)')
parser.add_argument('--model', default='pp_lidar_random', help='specify model (e.g.: pointpillar_5_aug_scale_augrev)')

args = parser.parse_args()

# First training stage on the full lidar point cloud
datasplit = '100'
os.system('cd tools && python train.py --cfg_file cfgs/' + args.folder + '/' + args.model + datasplit + '.yaml --max_ckpt_save_num 125 --fix_random_seed')

# Get Epoch with the best validation results from Tensorboard
event_acc = EventAccumulator('output/' + args.folder  + '/' + args.model + datasplit + '/default/tensorboard')
event_acc.Reload()
w_times, step_nums, vals = zip(*event_acc.Scalars('val/mAP_3d'))
Best_Epoch = vals.index(max(vals)) + 1

# Delete all other (non best) Epochs
ckptfiles = os.listdir('output/' + args.folder + '/' + args.model + datasplit + '/default/ckpt')
ckptfiles.remove('checkpoint_epoch_' + str(Best_Epoch) + '.pth')
for ckptfi in ckptfiles:
    os.remove('output/' + args.folder + '/' + args.model + datasplit + '/default/ckpt/' + ckptfi)

# Execute train of consecutive stages
datalist = ['100', '50', '25', '125', '625', '00'] # Definition of the stages to utilize in the training (00 is the final stage with training on just radar data)
for idx in range(datalist.__len__()):
    if idx == 0:
        continue
    datasplit = datalist[idx]
    os.system('cd tools && python train.py --cfg_file cfgs/' + args.folder + '/' + args.model + datasplit + '.yaml --ckpt' + ' ../output/' + args.folder + '/' + args.model + datalist[idx-1] + '/default/ckpt/' + 'checkpoint_epoch_' + str(Best_Epoch) + '.pth' + ' --max_ckpt_save_num 125 --fix_random_seed')

    #Get Epoch with the best validation results from Tensorboard
    event_acc = EventAccumulator('output/' + args.folder  + '/' + args.model + datasplit + '/default/tensorboard')
    event_acc.Reload()
    w_times, step_nums, vals = zip(*event_acc.Scalars('val/mAP_3d'))
    Best_Epoch = vals.index(max(vals)) + 1 + Best_Epoch

    # Delete all other (non best) Epochs
    ckptfiles = os.listdir('output/' + args.folder + '/' + args.model + datasplit + '/default/ckpt')
    ckptfiles.remove('checkpoint_epoch_' + str(Best_Epoch) + '.pth')
    for ckptfi in ckptfiles:
        os.remove('output/' + args.folder + '/' + args.model + datasplit + '/default/ckpt/' + ckptfi)

