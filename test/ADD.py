

import pdb
import time
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np


def get_critical_points(model, pc, label, num, text_features):

    B = label.shape[0]
    input_pc = pc.clone().detach().float().cuda()
    input_pc.requires_grad_()
    label = label.long().cuda()
    model.eval()


    point_features = model.encode_pc(input_pc)
    point_features = F.normalize(point_features, dim=-1)
    logits = torch.matmul(point_features, text_features.T)

    loss = F.cross_entropy(logits, label)
    loss.backward()

    with torch.no_grad():
        grad = input_pc.grad.data
        grad = torch.sum(grad ** 2, dim=-1)
        _, idx = grad.topk(k=num, dim=-1)
        critical_points = torch.stack([
            pc[i, idx[i]] for i in range(B)
        ], dim=0).clone().detach()
    return critical_points


class CWAdd:


    def __init__(self, model, adv_func, dist_func, text_features,
                 attack_lr=1e-2, init_weight=5e3, max_weight=4e4,
                 binary_step=10, num_iter=500, num_add=512):

        self.model = model.cuda().eval()
        self.adv_func = adv_func
        self.dist_func = dist_func
        self.text_features = text_features.cuda()
        self.attack_lr = attack_lr
        self.init_weight = init_weight
        self.max_weight = max_weight
        self.binary_step = binary_step
        self.num_iter = num_iter
        self.num_add = num_add

    def attack(self, data, target):

        B, K, _ = data.shape
        data = data.float().cuda().detach()
        ori_data = data.clone().detach()
        target = target.long().cuda().detach()
        label_val = target.cpu().numpy()

        lower_bound = np.zeros((B,))
        upper_bound = np.ones((B,)) * self.max_weight
        current_weight = np.ones((B,)) * self.init_weight

        o_bestdist = np.array([1e10] * B)
        o_bestscore = np.array([-1] * B)
        o_bestattack = np.zeros((B, self.num_add, 6))

        cri_data = get_critical_points(
            self.model, ori_data, target, self.num_add, self.text_features)

        for binary_step in range(self.binary_step):
            adv_data = cri_data + torch.randn((B, self.num_add, 6)).cuda() * 1e-7
            adv_data.requires_grad_()

            bestdist = np.array([1e10] * B)
            bestscore = np.array([-1] * B)

            opt = optim.Adam([adv_data], lr=self.attack_lr)

            for iteration in range(self.num_iter):

                cat_data = torch.cat([ori_data, adv_data], dim=1)
                point_features = self.model.encode_pc(cat_data)
                point_features = F.normalize(point_features, dim=-1)
                logits = torch.matmul(point_features, self.text_features.T)

                pred = torch.argmax(logits, dim=-1)

                if iteration % (self.num_iter // 5) == 0:
                    success_num = (pred == target).sum().item()
                    print(f"Step {binary_step}, iteration {iteration}, success {success_num}/{B}")

                dist_val = self.dist_func(adv_data, ori_data, batch_avg=False).detach().cpu().numpy()
                pred_val = pred.detach().cpu().numpy()
                input_val = adv_data.detach().cpu().numpy()

                for e, (dist, pred_label, label, ii) in \
                        enumerate(zip(dist_val, pred_val, label_val, input_val)):
                    if dist < bestdist[e] and pred_label == label:
                        bestdist[e] = dist
                        bestscore[e] = pred_label
                    if dist < o_bestdist[e] and pred_label == label:
                        o_bestdist[e] = dist
                        o_bestscore[e] = pred_label
                        o_bestattack[e] = ii

                adv_loss = self.adv_func(logits, target).mean()
                dist_loss = self.dist_func(
                    adv_data, ori_data,
                    weights=torch.from_numpy(current_weight).float().cuda()).mean()
                loss = adv_loss + dist_loss

                opt.zero_grad()
                loss.backward()
                opt.step()

                if iteration % 100 == 0:
                    torch.cuda.empty_cache()

            for e, label in enumerate(label_val):
                if bestscore[e] == label and bestscore[e] != -1 and bestdist[e] <= o_bestdist[e]:
                    lower_bound[e] = max(lower_bound[e], current_weight[e])
                else:
                    upper_bound[e] = min(upper_bound[e], current_weight[e])
                current_weight[e] = (lower_bound[e] + upper_bound[e]) / 2.

            torch.cuda.empty_cache()

        fail_idx = (lower_bound == 0.)
        o_bestattack[fail_idx] = input_val[fail_idx]

        success_num = (lower_bound > 0.).sum()
        print(f'Successfully attacked {success_num}/{B}')

        ori_data = ori_data.detach().cpu().numpy()
        final_attack = np.concatenate([ori_data, o_bestattack], axis=1)
        return o_bestdist, final_attack, success_num
