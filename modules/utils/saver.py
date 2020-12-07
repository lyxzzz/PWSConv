import torch
import os

class Saver():
    def __init__(self, cfg, logger, prefix = "epoch"):
        self.saver_cfg = cfg.pop("saver")
        self.interval = self.saver_cfg["interval"]
        self.prefix = prefix
        self.workdir = logger.workdir

    def save(self, epoch, model, optimizer):
        epoch = epoch + 1
        if epoch % self.interval == 0:
            save_name = f"{self.workdir}/{self.prefix}_{epoch}.pth"
            end_name = f"{self.workdir}/{self.prefix}_end.pth"
            print(f'Saving to {save_name}')
            state = {
                'state_dict': model.state_dict(),
                'epoch': epoch,
                'optimizer': optimizer.state_dict(),
            }
            torch.save(state, save_name)

            if os.path.islink(end_name):
                os.remove(end_name)

            os.symlink(os.path.basename(save_name), end_name)
            