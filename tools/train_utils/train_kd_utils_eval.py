import glob
import os

import torch
import tqdm
import time
from pcdet.utils import common_utils, commu_utils
from pcdet.config import cfg
from pcdet.models import load_data_to_gpu
from .train_utils import save_checkpoint, checkpoint_state
from .optimization import build_optimizer, build_scheduler
from pcdet.utils.kd_utils import kd_forwad
from tools.eval_utils import eval_utils_train


def train_one_epoch(model, optimizer, train_loader, model_func, lr_scheduler, accumulated_iter, optim_cfg,
                    rank, tbar, total_it_each_epoch, dataloader_iter, tb_log=None, leave_pbar=False, teacher_model=None,
                    extra_optim=None, extra_lr_scheduler=None):

    accumulated_loss = 0

    if total_it_each_epoch == len(train_loader):
        dataloader_iter = iter(train_loader)

    forward_func = getattr(kd_forwad, cfg.KD.get('FORWARD_FUNC', 'forward'))

    if rank == 0:
        pbar = tqdm.tqdm(total=total_it_each_epoch, leave=leave_pbar, desc='train', dynamic_ncols=True)
        data_time = common_utils.AverageMeter()
        batch_time = common_utils.AverageMeter()

    for cur_it in range(total_it_each_epoch):
        end = time.time()
        try:
            batch = next(dataloader_iter)
        except StopIteration:
            dataloader_iter = iter(train_loader)
            batch = next(dataloader_iter)
            print('new iters')
        
        data_timer = time.time()
        cur_data_time = data_timer - end

        lr_scheduler.step(accumulated_iter)
        if extra_lr_scheduler is not None:
            extra_lr_scheduler.step(accumulated_iter)

        try:
            cur_lr = float(optimizer.lr)
        except:
            cur_lr = optimizer.param_groups[0]['lr']

        if tb_log is not None:
            tb_log.add_scalar('meta_data/learning_rate', cur_lr, accumulated_iter)
            if extra_optim is not None:
                tb_log.add_scalar('meta_data/extra_lr', float(extra_optim.lr), accumulated_iter)

        model.train()
        
        loss, tb_dict, disp_dict = forward_func(
            model, teacher_model, batch, optimizer, extra_optim, optim_cfg, load_data_to_gpu
        )

        accumulated_loss += loss.item()
        accumulated_iter += 1

        cur_batch_time = time.time() - end
        # average reduce
        avg_data_time = commu_utils.average_reduce_value(cur_data_time)
        avg_batch_time = commu_utils.average_reduce_value(cur_batch_time)

        if cur_it % 20 == 0:
            torch.cuda.empty_cache()

        # log to console and tensorboard
        if rank == 0:
            data_time.update(avg_data_time)
            batch_time.update(avg_batch_time)
            disp_dict.update({
                'label_loss': tb_dict['label_loss'],
                'loss': loss.item(), 'lr': cur_lr, 'd_t': f'({data_time.avg:.1f})',
                'b_t': f'{batch_time.val:.1f}({batch_time.avg:.1f})'
            })

            pbar.update()
            pbar.set_postfix(dict(total_it=accumulated_iter))
            tbar.set_postfix(disp_dict)
            tbar.refresh()

            if tb_log is not None:
                tb_log.add_scalar('train/loss', loss, accumulated_iter)
                tb_log.add_scalar('meta_data/learning_rate', cur_lr, accumulated_iter)
                for key, val in tb_dict.items():
                    tb_log.add_scalar('train/' + key, val, accumulated_iter)
    if rank == 0:
        pbar.close()
    accumulated_loss = accumulated_loss / len(train_loader)

    return accumulated_iter, accumulated_loss

def train_model_kd(model, optimizer, train_loader, test_loader, model_func, lr_scheduler, optim_cfg,
                   start_epoch, total_epochs, start_iter, rank, tb_log, ckpt_save_dir, train_sampler=None,
                   lr_warmup_scheduler=None, ckpt_save_interval=1, max_ckpt_save_num=50,
                   merge_all_iters_to_one_epoch=False, teacher_model=None):
    extra_optim = extra_lr_scheduler = None
    if optim_cfg.get('EXTRA_OPTIM', None) and optim_cfg.EXTRA_OPTIM.ENABLED:
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            extra_optim = build_optimizer(model.module, optim_cfg.EXTRA_OPTIM)
        else:
            extra_optim = build_optimizer(model, optim_cfg.EXTRA_OPTIM)

        # last epoch is no matter for one cycle scheduler
        extra_lr_scheduler, _ = build_scheduler(
            extra_optim, total_iters_each_epoch=len(train_loader), total_epochs=total_epochs,
            last_epoch=-1, optim_cfg=optim_cfg.EXTRA_OPTIM
        )

    accumulated_iter = start_iter

    if teacher_model is not None and cfg.KD.get('TEACHER_MODE', None):
        getattr(teacher_model, cfg.KD.TEACHER_MODE)()

    if teacher_model is not None and cfg.KD.get('TEACHER_BN_MODE', None) == 'train':
        teacher_model.apply(common_utils.set_bn_train)

    with tqdm.trange(start_epoch, total_epochs, desc='epochs', dynamic_ncols=True, leave=(rank == 0)) as tbar:
        total_it_each_epoch = len(train_loader)
        if merge_all_iters_to_one_epoch:
            assert hasattr(train_loader.dataset, 'merge_all_iters_to_one_epoch')
            train_loader.dataset.merge_all_iters_to_one_epoch(merge=True, epochs=total_epochs)
            total_it_each_epoch = len(train_loader) // max(total_epochs, 1)

        dataloader_iter = iter(train_loader)
        for cur_epoch in tbar:
            if cur_epoch == 100:
                break
            if train_sampler is not None:
                train_sampler.set_epoch(cur_epoch)

            # train one epoch
            if lr_warmup_scheduler is not None and cur_epoch < optim_cfg.WARMUP_EPOCH:
                cur_scheduler = lr_warmup_scheduler
            else:
                cur_scheduler = lr_scheduler
            accumulated_iter, accumulated_loss = train_one_epoch(
                model, optimizer, train_loader, model_func,
                lr_scheduler=cur_scheduler,
                accumulated_iter=accumulated_iter, optim_cfg=optim_cfg,
                rank=rank, tbar=tbar, tb_log=tb_log,
                leave_pbar=(cur_epoch + 1 == total_epochs),
                total_it_each_epoch=total_it_each_epoch,
                dataloader_iter=dataloader_iter,
                teacher_model=teacher_model,
                extra_optim=extra_optim,
                extra_lr_scheduler=extra_lr_scheduler
            )

            #TODO: Calculate Additional Losses on the Validation Dataset
            ret = eval_utils_train.eval_one_epoch_train(model, test_loader)
            print(ret['mAP_3d'])
            #session.report({"loss": accumulated_loss, "mAP_3d": ret['mAP_3d'], "mAP_bev": ret['mAP_bev']})
            # tune.report(loss=accumulated_loss)

            # ret_train = eval_utils_train.eval_one_epoch_train(model, train_loader)

            # Add Information to Tensorboard
            tb_log.add_scalar('train/epochloss', accumulated_loss, cur_epoch)
            tb_log.add_scalar('val/car_bev', ret['Car_bev/easy_R40'], cur_epoch)
            tb_log.add_scalar('val/car_3d', ret['Car_3d/easy_R40'], cur_epoch)

            tb_log.add_scalar('val/mAP_3d', ret['mAP_3d'], cur_epoch)
            tb_log.add_scalar('val/mAP_bev', ret['mAP_bev'], cur_epoch)

            try:
                tb_log.add_scalar('val/pedestrain_bev', ret['Pedestrian_bev/easy'], cur_epoch)
                tb_log.add_scalar('val/pedestrain_3d', ret['Pedestrian_3d/easy'], cur_epoch)
                tb_log.add_scalar('val/cyclist_bev', ret['Cyclist_bev/easy'], cur_epoch)
                tb_log.add_scalar('val/cyclist_3d', ret['Cyclist_3d/easy'], cur_epoch)
            except:
                pass

            # # Add Information to Tensorboard
            # tb_log.add_scalar('train/car_bev', ret_train['Car_bev/easy_R40'], cur_epoch)
            # tb_log.add_scalar('train/car_3d', ret_train['Car_3d/easy_R40'], cur_epoch)
            #
            # tb_log.add_scalar('train/mAP_3d', ret_train['mAP_3d'], cur_epoch)
            # tb_log.add_scalar('train/mAP_bev', ret_train['mAP_bev'], cur_epoch)
            #
            # try:
            #     tb_log.add_scalar('train/pedestrain_bev', ret_train['Pedestrian_bev/easy'], cur_epoch)
            #     tb_log.add_scalar('train/pedestrain_3d', ret_train['Pedestrian_3d/easy'], cur_epoch)
            #     tb_log.add_scalar('train/cyclist_bev', ret_train['Cyclist_bev/easy'], cur_epoch)
            #     tb_log.add_scalar('train/cyclist_3d', ret_train['Cyclist_3d/easy'], cur_epoch)
            # except:
            #     pass

            # save trained model
            trained_epoch = cur_epoch + 1
            if trained_epoch % ckpt_save_interval == 0 and rank == 0:

                ckpt_list = glob.glob(str(ckpt_save_dir / 'checkpoint_epoch_*.pth'))
                ckpt_list.sort(key=os.path.getmtime)

                if ckpt_list.__len__() >= max_ckpt_save_num:
                    for cur_file_idx in range(0, len(ckpt_list) - max_ckpt_save_num + 1):
                        os.remove(ckpt_list[cur_file_idx])

                ckpt_name = ckpt_save_dir / ('checkpoint_epoch_%d' % trained_epoch)
                save_checkpoint(
                    checkpoint_state(model, optimizer, trained_epoch, accumulated_iter), filename=ckpt_name,
                )

