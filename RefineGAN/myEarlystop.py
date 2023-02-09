import numpy as np
import torch
class EarlyStopping:
    def __init__(self , patience , savepath, cpname, verbose = False , delta = 0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False        
        self.val_loss_min = np.Inf
        self.delta = delta
        self.cpname = cpname
        self.savepath = savepath
    def __call__(self , val_loss , model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model, self.savepath+self.cpname+'.pth')	# 这里会存储迄今最优模型的参数
        self.val_loss_min = val_loss
