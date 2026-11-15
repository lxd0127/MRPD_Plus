

import pdb
import time
import torch
import torch.optim as optim
import numpy as np


class CWPerturb:


    def __init__(self, model, adv_func, dist_func, attack_lr=1e-2,
                 init_weight=10., max_weight=80., binary_step=10, num_iter=500,text_features = None):


        self.model = model.cuda()
        self.model.eval()
        self.text_features = text_features
        self.adv_func = adv_func
        self.dist_func = dist_func
        self.attack_lr = attack_lr
        self.init_weight = init_weight
        self.max_weight = max_weight
        self.binary_step = binary_step
        self.num_iter = num_iter

    def attack(self, data, target):

        B, K = data.shape[:2]
        data = data.float().cuda().detach()

        ori_data = data.clone().detach()
        ori_data.requires_grad = False
        target = target.long().cuda().detach()
        label_val = target.detach().cpu().numpy()


        lower_bound = np.zeros((B,))
        upper_bound = np.ones((B,)) * self.max_weight
        current_weight = np.ones((B,)) * self.init_weight


        o_bestdist = np.array([1e10] * B)
        o_bestscore = np.array([-1] * B)
        o_bestattack = np.zeros((B, K, 6))


        for binary_step in range(self.binary_step):

            adv_data = ori_data.clone().detach() + \
                torch.randn((B, K, 6)).cuda() * 1e-7
            adv_data.requires_grad_()
            bestdist = np.array([1e10] * B)
            bestscore = np.array([-1] * B)
            opt = optim.Adam([adv_data], lr=self.attack_lr, weight_decay=0.)

            adv_loss = torch.tensor(0.).cuda()
            dist_loss = torch.tensor(0.).cuda()

            total_time = 0.
            forward_time = 0.
            backward_time = 0.
            update_time = 0.


            for iteration in range(self.num_iter):
                t1 = time.time()


                point_features = self.model.encode_pc(adv_data)
                point_features = torch.nn.functional.normalize(point_features, dim=-1)
                logits = torch.matmul(point_features, self.text_features.T)

                if isinstance(logits, tuple):
                    logits = logits[0]

                t2 = time.time()
                forward_time += t2 - t1


                pred = torch.argmax(logits, dim=1)
                success_num = (pred == target).sum().item()
                if iteration % (self.num_iter // 5) == 0:
                    print('Step {}, iteration {}, success {}/{}\n'
                          'adv_loss: {:.4f}, dist_loss: {:.4f}'.
                          format(binary_step, iteration, success_num, B,
                                 adv_loss.item(), dist_loss.item()))


                dist_val = torch.sqrt(torch.sum(
                    (adv_data - ori_data) ** 2, dim=[1, 2])).\
                    detach().cpu().numpy()
                pred_val = pred.detach().cpu().numpy()
                input_val = adv_data.detach().cpu().numpy()


                for e, (dist, pred, label, ii) in \
                        enumerate(zip(dist_val, pred_val, label_val, input_val)):
                    if dist < bestdist[e] and pred == label:
                        bestdist[e] = dist
                        bestscore[e] = pred
                    if dist < o_bestdist[e] and pred == label:
                        o_bestdist[e] = dist
                        o_bestscore[e] = pred
                        o_bestattack[e] = ii

                t3 = time.time()
                update_time += t3 - t2


                adv_loss = self.adv_func(logits, target).mean()
                dist_loss = self.dist_func(adv_data, ori_data,
                                           torch.from_numpy(
                                               current_weight)).mean()
                loss = adv_loss + dist_loss
                opt.zero_grad()
                loss.backward()
                opt.step()

                t4 = time.time()
                backward_time += t4 - t3
                total_time += t4 - t1

                if iteration % 100 == 0:
                    print('total time: {:.2f}, for: {:.2f}, '
                          'back: {:.2f}, update: {:.2f}'.
                          format(total_time, forward_time,
                                 backward_time, update_time))
                    total_time = 0.
                    forward_time = 0.
                    backward_time = 0.
                    update_time = 0.
                    torch.cuda.empty_cache()


            for e, label in enumerate(label_val):
                if bestscore[e] == label and bestscore[e] != -1 and bestdist[e] <= o_bestdist[e]:

                    lower_bound[e] = max(lower_bound[e], current_weight[e])
                    current_weight[e] = (lower_bound[e] + upper_bound[e]) / 2.
                else:

                    upper_bound[e] = min(upper_bound[e], current_weight[e])
                    current_weight[e] = (lower_bound[e] + upper_bound[e]) / 2.

            torch.cuda.empty_cache()


        fail_idx = (lower_bound == 0.)
        o_bestattack[fail_idx] = input_val[fail_idx]


        success_num = (lower_bound > 0.).sum()
        print('Successfully attack {}/{}'.format(success_num, B))
        return o_bestdist, o_bestattack, success_num
