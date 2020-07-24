import numpy as np
import torch
import torch.nn as nn



class EagerPruner(object):
    def __init__(self, model, 
                 epoch_num,
                 each_epoch_iters,
                 prune_interval=5000, 
                 prune_num=20000, 
                 over_prune_threshold=20, 
                 prune_fail_times=3,
                 max_prune_rate=0.85,
                 beishu=1,
                 force_flag=False,
                 min_prune_rate=0.5,
                 check_beishu=0.4,
                 max_beishu=0.75):
        self.model = model
        self.finish_flag = False

        # 超参数
        self.prune_interval = prune_interval
        self.prune_num = prune_num
        self.over_prune_threshold = over_prune_threshold
        self.prune_fail_times = prune_fail_times
        self.max_prune_rate = max_prune_rate

        self.beishu = beishu
        self.force_flag=force_flag
        self.min_prune_rate=min_prune_rate
        self.check_beishu = check_beishu
        self.max_beishu = max_beishu

        ''' 
        辅助判断是否到了剪枝的时刻
        epoch_num:训练过程中的epoch总数
        each_epoch_iters:每一个epoch中的迭代次数
        last_rb_iter: 上一次发生剪枝所在的迭代时刻
        超参数prune_interval与此部分有关

        '''
        self.epoch_num = epoch_num
        self.each_epoch_iters = each_epoch_iters
        self.last_rb_iter = 0
        
        '''
        辅助判断剪枝过后，窗口内的平均loss是否超过上一个剪枝周期
        window_size:滑动窗口大小
        window_list:存放loss，方便求smoothed_loss
        last_prune_loss_max:上一个简直周期中出现的最大的smoothed_loss
        curr_prune_loss_max:当前周期中到目前为止出现的最大smoothed_loss
        curr_exceed_times:到目前为止超过last_prune_loss_max的次数
        超参数over_prune_threshold与此部分有关

        '''
        self.window_size = 100
        self.window_list = [0] * self.window_size
        self.last_prune_loss_max = float('inf')
        self.curr_prune_loss_max = 0
        self.curr_exceed_times = 0
        self.curr_idx = 0
        self.thres_idx_for_loss = min(600, int(self.prune_interval * self.check_beishu))
        


        '''
        辅助判断连续失败次数，判断是否需要终止剪枝
        curr_fail_times:当前失败的次数
        roll_back_flag: 上一次剪枝到现在是否发生了回滚
        超参数prune_fail_times,max_prune_rate与此部分有关
        '''
        self.curr_fail_times = 0
        self.roll_back_flag = False


        '''
        curr_num记录当前已裁剪的权重总数
        记录所有需要裁减的权重的总数，每个要裁减的层的size
        为每个层创建mask矩阵
        '''
        self.curr_num = 0
        self.all_weights_num = 0
        self.layer_size_list = []
        self.mask_list = []
        
        '''
        备份权重以及权重的mask矩阵
        '''
        self.backup_weight_list = []
        self.backup_mask_list = []
        for layer in self.model.modules():
            if isinstance(layer, nn.Conv2d):
                self.layer_size_list.append(layer.weight.data.shape)
                self.all_weights_num += np.prod([s for s in layer.weight.data.shape])
                self.mask_list.append(torch.ones(layer.weight.data.shape))
                self.backup_weight_list.append(layer.weight.data.clone().detach())
                self.backup_mask_list.append(torch.ones(layer.weight.data.shape))

        
        
    def get_prune_rate(self):
        overall_rate = self.curr_num / self.all_weights_num
        layers_rate = [torch.mean((self.mask_list[i] == 0).float()).item() for i in range(len(self.mask_list))]
        return overall_rate, layers_rate
        
    def set_zero_by_mask(self):
        if self.curr_num == 0:
            return
        i = 0
        with torch.no_grad():
            for layer in self.model.modules():
                if isinstance(layer, nn.Conv2d):
                    mask = self.mask_list[i]
                    i += 1
                    layer.weight.data[mask == 0] = 0
    
    def pruning_check(self, curr_epoch, curr_iter):
        return (curr_epoch * self.each_epoch_iters + curr_iter - self.last_rb_iter) == self.prune_interval
    
    def backup_and_prune(self, curr_epoch, curr_iter):
        # 如果此次裁减后要超过了一定的比率，就不再进行裁减
        if self.force_flag and \
           curr_epoch <= int(self.epoch_num * 0.55) and \
           curr_epoch >= int(self.epoch_num * 0.40) and \
           self.curr_num < int(self.all_weights_num * self.min_prune_rate):
            remain_num = self.all_weights_num * self.max_prune_rate - self.curr_num
            remain_iters = (self.epoch_num - curr_epoch) * self.each_epoch_iters - curr_iter
            self.prune_num = int(remain_num * self.prune_interval / remain_iters)
        num_need_sort = self.curr_num + self.prune_num
        if num_need_sort > int(self.all_weights_num * self.max_prune_rate) or curr_epoch >= int(self.epoch_num * self.max_beishu):
            self.finish_flag=True
            return
        all_weights_flatten = []
        mask_flatten = torch.ones(self.all_weights_num)
        old_mask_flatten = []
        with torch.no_grad():
            i = 0
            for layer in self.model.modules():
                if isinstance(layer, nn.Conv2d):
                    self.backup_weight_list[i] = layer.weight.data.clone().detach()
                    self.backup_mask_list[i] = self.mask_list[i].clone().detach()
                    old_mask_flatten.append(self.mask_list[i].flatten())
                    i += 1
                    all_weights_flatten.append(layer.weight.data.flatten())

            all_weights_flatten = np.abs(torch.cat(all_weights_flatten).cpu().numpy())
            old_mask_flatten = torch.cat(old_mask_flatten).cpu().numpy()
            assert np.sum(old_mask_flatten == 0) == self.curr_num
            all_weights_flatten[old_mask_flatten == 0] = -1.0
            smallest_idx = np.argpartition(all_weights_flatten, num_need_sort - 1)[:num_need_sort]
            mask_flatten[smallest_idx] = 0
            start_idx = 0
            for i, torch_size in enumerate(self.layer_size_list):
                end_idx = start_idx + np.prod([s for s in torch_size])
                mask = mask_flatten[start_idx:end_idx].view(torch_size)
                self.mask_list[i] = mask.clone().detach()
                start_idx = end_idx
        self.set_zero_by_mask()

        '''
        如果没有发生回滚，那么将连续失败次数重置为0
        '''
        if self.roll_back_flag == False:
            self.curr_fail_times = 0
        self.last_prune_loss_max = self.curr_prune_loss_max
        self.curr_prune_loss_max = 0
        self.curr_exceed_times = 0
        self.curr_idx = 0
        self.last_rb_iter = curr_epoch * self.each_epoch_iters + curr_iter

        # 滑动窗口是否要清零待定,先考虑不清零
        # self.window_list = [0] * self.window_size
        self.curr_num = num_need_sort
        self.roll_back_flag = False
        return 
    
    def loss_increase_check(self, curr_loss):
        first_ele = self.window_list.pop(0)
        self.window_list.append(curr_loss)
#         self.smoothed_loss += (curr_loss - first_ele) / self.window_size

        if self.curr_idx >= self.thres_idx_for_loss:
            smoothed_loss = np.mean(np.array(self.window_list))
            if smoothed_loss > self.curr_prune_loss_max:
                self.curr_prune_loss_max = smoothed_loss
            if smoothed_loss > self.last_prune_loss_max * self.beishu:
                self.curr_exceed_times += 1
                return True
        self.curr_idx += 1
        return False
        
    def roll_back_check(self):
        return self.curr_exceed_times > self.over_prune_threshold
    
    def roll_back(self, curr_epoch, curr_iter):
        i = 0
        with torch.no_grad():
            for layer in self.model.modules():
                if isinstance(layer, nn.Conv2d):
                    layer.weight.data = self.backup_weight_list[i]
                    self.mask_list[i] = self.backup_mask_list[i]
                    i += 1
        self.curr_num  -= self.prune_num
        self.roll_back_flag = True
        self.curr_exceed_times = 0
        self.curr_fail_times += 1
        self.curr_idx = 0
        self.last_prune_loss_max = float('inf')
        self.curr_prune_loss_max = 0

        if self.force_flag:
            self.over_prune_threshold = int(self.over_prune_threshold * 1.25)
            over_prune_max = int(0.75*(self.prune_interval * self.check_beishu))
            self.over_prune_threshold = min(self.over_prune_threshold, over_prune_max)
        
        self.last_rb_iter = curr_epoch * self.each_epoch_iters + curr_iter
        
        return
    
    def pruning_termination_check(self):
        return (self.curr_fail_times > self.prune_fail_times)
        
    
        
