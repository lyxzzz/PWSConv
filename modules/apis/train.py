import torch
import torch.nn as nn
import time

def epoch_train(model:nn.Module, epoch, device, data_loader, optimizer, scheduler, logger, saver):
    model.train()
    it = 0
    total_iter = data_loader.__len__()
    start_time = time.time()
    logger.new_epoch()
    for batch_idx, (inputs, targets) in enumerate(data_loader):
        it += 1
        inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        loginfo = {
            'mode': "train",
            'datatime': round(time.time() - start_time, 4),
            'epoch': epoch,
            'iter': it,
            'lr': optimizer.param_groups[0]['lr'],
            'total': total_iter,
            'batchsize': targets.size(0)
        }

        optimizer.zero_grad()

        outputs = model(inputs, targets)
        loss = outputs['loss']

        loss.backward()
        optimizer.step()

        loginfo['time'] = round(time.time() - start_time, 4)
        start_time = loginfo['time'] + start_time
        logger.record_train(loginfo, outputs)
        
    scheduler.step()
    saver.save(epoch, model, optimizer)