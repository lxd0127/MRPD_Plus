import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F

class CWKNN:


    def __init__(self, model, adv_func, dist_func, clip_func,
                 attack_lr=1e-3, num_iter=2500,text_features=None):
        if isinstance(model, torch.nn.DataParallel):
            self.model = model.module
        else:
            self.model = model
        self.model.cuda()
        self.model.eval()

        self.adv_func = adv_func
        self.dist_func = dist_func
        self.clip_func = clip_func
        self.attack_lr = attack_lr
        self.num_iter = num_iter
        self.text_features = text_features

    def attack(self, data, target):

        B, K, C = data.shape
        assert C >= 3, "Expected at least xyz coordinates in input."

        data = data.float().cuda().detach()
        target = target.long().cuda().detach()


        ori_xyz = data[:, :, :3].clone().detach()
        other_feats = data[:, :, 3:].clone().detach() if C > 3 else None


        adv_xyz = ori_xyz.clone().detach() + torch.randn_like(ori_xyz) * 1e-7
        adv_xyz.requires_grad_()
        optimizer = optim.Adam([adv_xyz], lr=self.attack_lr)

        for iteration in range(self.num_iter):

            if other_feats is not None:
                adv_input = torch.cat([adv_xyz, other_feats], dim=2)
            else:
                adv_input = adv_xyz


            point_features = self.model.encode_pc(adv_input)
            point_features = F.normalize(point_features, dim=-1)
            logits = torch.matmul(point_features, self.text_features.T)


            if isinstance(logits, tuple):
                logits = logits[0]

            pred = torch.argmax(logits, dim=1)
            success_num = (pred == target).sum().item()

            if iteration % (self.num_iter // 5) == 0:
                print(f'Iter {iteration}/{self.num_iter}, success {success_num}/{B}')


            adv_loss = self.adv_func(logits, target).mean()
            dist_loss = self.dist_func(adv_xyz, ori_xyz).mean() * K
            loss = adv_loss + dist_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


            adv_xyz.data = self.clip_func(adv_xyz.clone().detach(), ori_xyz, normal=None)

            if iteration % 100 == 0:
                torch.cuda.empty_cache()


        '''with torch.no_grad():
            if other_feats is not None:
                adv_input = torch.cat([adv_xyz, other_feats], dim=2)
            else:
                adv_input = adv_xyz

            logits = self.model(adv_input)
            if isinstance(logits, tuple):
                logits = logits[0]
            pred = torch.argmax(logits, dim=-1)
            success_num = (pred == target).sum().detach().cpu().item()

        print(f'Successfully attacked {success_num}/{B} samples.')'''


        adv_input = adv_input.detach().cpu().numpy()
        return adv_input,0
