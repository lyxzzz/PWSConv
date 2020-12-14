import time
import os
import shutil
import logging
import json

class Logger():

    def __init__(self, args, cfg, save_file = True):

        logging.basicConfig(level = logging.INFO, format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger("myselfsup")
        self.logger.setLevel(logging.INFO)

        if save_file:
            if args.workdir == "":
                self.workdir = "result/" + os.path.basename(args.config)[:-3]
            else:
                self.workdir = args.workdir
        
            if not os.path.isdir(self.workdir):
                os.makedirs(self.workdir)
            timestamp = time.strftime("%Y%m%d_%H%M%S.log", time.localtime())
            fh = logging.FileHandler(f"{self.workdir}/train_{timestamp}", mode='w')
            fh.setLevel(logging.INFO)
            fh.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
            self.logger.addHandler(fh)
        
            self.jsonfile = open(f'{self.workdir}/{timestamp}.json', "w")
            shutil.copy(args.config, self.workdir)

            self.logger_cfg = cfg.pop("logger")
            self.interval = self.logger_cfg["interval"]
            self.total_epoch = cfg["total_epochs"]
            self.train_outputs = None
        else:
            self.jsonfile = None
        # self.now = 
    
    def print(self, msg):
        self.logger.info(msg)

    def _cal_eta(self, epoch, it, total_iter, t):
        left_iter = (self.total_epoch - epoch - 1) * total_iter + total_iter - it
        left_sec = int(left_iter * t)

        left_min = int(left_sec / 60)
        left_sec = left_sec % 60

        left_hour = int(left_min / 60)
        left_min = left_min % 60

        left_day = int(left_hour / 24)
        left_hour = left_hour % 24
        
        if left_day > 0:
            result = '{} day, {:02}:{:02}:{:02}'.format(left_day, left_hour, left_min, left_sec)
        else:
            result = '{:02}:{:02}:{:02}'.format(left_hour, left_min, left_sec)
        
        return result


    def record_train(self, loginfo, outputs):
        if self.train_outputs is None:
            self.train_outputs = {}
            for entry, value in outputs.items():
                if isinstance(value, dict):
                    self.train_outputs[entry] = {}
                    for k, v in value.items():
                        self.train_outputs[entry][k] = v.item()
                else:
                    self.train_outputs[entry] = value.item()

            self.train_outputs['sample_num'] = loginfo['batchsize']
            self.train_outputs['time'] = loginfo['time']
            self.train_outputs['datatime'] = loginfo['datatime']
        else:
            for k1, v1 in outputs.items():
                if isinstance(v1, dict):
                    for k2, v2 in v1.items():
                        self.train_outputs[k1][k2] += v2.item()
                else:
                    self.train_outputs[k1] += v1.item()
            
            self.train_outputs['sample_num'] += loginfo['batchsize']
            self.train_outputs['time'] += loginfo['time']
            self.train_outputs['datatime'] += loginfo['datatime']


        if loginfo["iter"] % self.interval == 0:
            loginfo['time'] = self.train_outputs.pop('time') / self.interval
            loginfo['datatime'] = self.train_outputs.pop('datatime') / self.interval
            eta_time = self._cal_eta(loginfo['epoch'], loginfo['iter'], loginfo['total'], loginfo['time'])
            loginfo['time'] = "{:.3f}".format(loginfo['time'])
            loginfo['datatime'] = "{:.3f}".format(loginfo['datatime'])
            loginfo['lr'] = "{:.2e}".format(loginfo['lr'])

            msg = "Epoch [{}][{}/{}]\tlr: {}, eta: {}, time: {}, data_time: {}".format(
                                                    loginfo['epoch'], loginfo['iter'], loginfo['total'], 
                                                    loginfo['lr'], eta_time,
                                                    loginfo['time'], loginfo['datatime'])
            
            batchsize = loginfo['batchsize']
            sample_nums = self.train_outputs.pop('sample_num')
            for k1, v1 in self.train_outputs.items():
                if isinstance(v1, dict):
                    for k2, v2 in v1.items():
                        loginfo[k2] = round(v2 * (100.0 / sample_nums), 4)
                        msg = "{}, {}: {:.4f}".format(msg, k2, loginfo[k2])
                else:
                    loginfo[k1] = round(v1 / float(self.interval), 4)
                    msg = "{}, {}: {:.4f}".format(msg, k1, loginfo[k1])
            
            self.train_outputs = None
            
            msg = "{}, batchsize: {}".format(msg, batchsize)
            self.print(msg)

            self.jsonfile.write("{}\n".format(json.dumps(loginfo)))
            self.jsonfile.flush()

    def record_eval(self, epoch, outputs):
        msg = "Epoch [{}]\t".format(epoch)
        loginfo = {"mode":"val", "epoch":epoch}

        for k in outputs:
            loginfo[k] = outputs[k]
            msg = "{}, {}: {:.4f}".format(msg, k, loginfo[k])
        self.print(msg)
        if self.jsonfile is not None:
            self.jsonfile.write("{}\n".format(json.dumps(loginfo)))
            self.jsonfile.flush()

    
    def new_epoch(self):
        self.train_outputs = None