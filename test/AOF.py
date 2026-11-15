

import pdb
import time
import torch
import torch.optim as optim
import numpy as np

def knn(x, k):

    with torch.no_grad():
        inner = -2 * torch.matmul(x.transpose(2, 1), x)
        xx = torch.sum(x ** 2, dim=1, keepdim=True)
        pairwise_distance = -xx - inner - xx.transpose(2, 1)

        vec = x.transpose(2, 1).unsqueeze(2) - x.transpose(2, 1).unsqueeze(1)
        dist = -torch.sum(vec**2, dim=-1)


        idx = pairwise_distance.topk(k=k, dim=-1)[1]
    return idx


def get_Laplace_from_pc(ori_pc):


    pc = ori_pc.detach().clone()
    with torch.no_grad():

        idx = knn(pc, 30)
        pc = pc.transpose(2, 1).contiguous()
        point_mat = pc.unsqueeze(2) - pc.unsqueeze(1)
        A = torch.exp(-torch.sum(point_mat.square(), dim=3))
        mask = torch.zeros_like(A)
        mask.scatter_(2, idx, 1)
        mask = mask + mask.transpose(2, 1)
        mask[mask>1] = 1

        A = A * mask
        D = torch.diag_embed(torch.sum(A, dim=2))
        L = D - A

        e, v = torch.linalg.eigh(L, UPLO='L')
    return e.to(ori_pc), v.to(ori_pc)


class CWAOF:


    def __init__(self, model, adv_func, dist_func, attack_lr=1e-2,
                 binary_step=2, num_iter=200, GAMMA=0.5, low_pass=100, clip_func=None, text_features = None):


        self.model = model.cuda()
        self.model.eval()
        self.text_features = text_features
        self.adv_func = adv_func
        self.dist_func = dist_func
        self.attack_lr = attack_lr
        self.binary_step = binary_step
        self.num_iter = num_iter
        self.GAMMA = GAMMA
        self.low_pass = low_pass
        self.clip_func = clip_func

    def attack(self, data, target):

        B, K = data.shape[:2]
        data = data.float().cuda().detach()
        data = data.transpose(1, 2).contiguous()
        ori_data = data.clone().detach()
        ori_data.requires_grad = False
        target = target.long().cuda().detach()
        label_val = target.detach().cpu().numpy()


        o_bestdist = np.array([1e10] * B)
        o_bestscore = np.array([-1] * B)
        o_bestattack = np.zeros((B, 3, K))


        for binary_step in range(self.binary_step):

            adv_data = ori_data.clone().detach() + \
                torch.randn((B, 3, K)).cuda() * 1e-7
            adv_data.requires_grad_(False)
            Evs, V = get_Laplace_from_pc(adv_data)
            projs = torch.bmm(adv_data, V)
            hfc = torch.bmm(projs[..., self.low_pass:],V[..., self.low_pass:].transpose(2, 1))
            lfc = torch.bmm(projs[..., :self.low_pass],V[..., :self.low_pass].transpose(2, 1))
            lfc = lfc.detach().clone()
            hfc = hfc.detach().clone()
            lfc.requires_grad_()
            hfc.requires_grad_(False)
            ori_lfc = lfc.detach().clone()
            ori_lfc.requires_grad_(False)
            ori_hfc = hfc.detach().clone()
            ori_hfc.requires_grad_(False)
            opt = optim.Adam([lfc], lr=self.attack_lr, weight_decay=0.)

            adv_loss = torch.tensor(0.).cuda()
            dist_loss = torch.tensor(0.).cuda()

            total_time = 0.
            optimize_time = 0.
            clip_time = 0.
            update_time = 0.


            for iteration in range(self.num_iter):
                t1 = time.time()


                adv_data = lfc + hfc


                point_features = self.model.encode_pc(torch.cat((adv_data.transpose(2, 1), torch.full_like(adv_data.transpose(2, 1), 0.5)), -1))
                point_features = torch.nn.functional.normalize(point_features, dim=-1)
                logits = torch.matmul(point_features, self.text_features.T)
                if isinstance(logits, tuple):
                    logits = logits[0]

                adv_loss = (1-self.GAMMA) * self.adv_func(logits, target).mean()
                opt.zero_grad()
                adv_loss.backward()


                point_features = self.model.encode_pc(torch.cat((lfc.transpose(2, 1), torch.full_like(lfc.transpose(2, 1), 0.5)), -1))
                point_features = torch.nn.functional.normalize(point_features, dim=-1)
                lfc_logits = torch.matmul(point_features, self.text_features.T)
                if isinstance(lfc_logits, tuple):
                    lfc_logits = lfc_logits[0]

                lfc_adv_loss = self.GAMMA * self.adv_func(lfc_logits, target).mean()
                lfc_adv_loss.backward()
                opt.step()

                t2 = time.time()
                optimize_time += t2 - t1


                with torch.no_grad():
                    adv_data = lfc + hfc
                    adv_data.data = self.clip_func(adv_data.detach().clone(), ori_data)
                    coeff = torch.bmm(adv_data, V)
                    hfc.data = torch.bmm(coeff[..., self.low_pass:],V[..., self.low_pass:].transpose(2, 1))
                    lfc.data = torch.bmm(coeff[..., :self.low_pass],V[..., :self.low_pass].transpose(2, 1))

                t3 = time.time()
                clip_time += t3 - t2


                with torch.no_grad():

                    point_features = self.model.encode_pc(torch.cat((adv_data.transpose(2, 1), torch.full_like(adv_data.transpose(2, 1), 0.5)), -1))
                    point_features = torch.nn.functional.normalize(point_features, dim=-1)
                    flogits = torch.matmul(point_features, self.text_features.T)
                    if isinstance(flogits, tuple):
                        flogits = flogits[0]
                    pred = torch.argmax(flogits, dim=1)


                    point_features = self.model.encode_pc(torch.cat((lfc.transpose(2, 1), torch.full_like(lfc.transpose(2, 1), 0.5)), -1))
                    point_features = torch.nn.functional.normalize(point_features, dim=-1)
                    lfc_flogits = torch.matmul(point_features, self.text_features.T)

                    if isinstance(lfc_flogits, tuple):
                        lfc_flogits = lfc_flogits[0]
                    lfc_pred = torch.argmax(lfc_flogits, dim=1)
                success_num = ((pred != target)*(lfc_pred != target)).sum().item()
                if iteration % (self.num_iter // 5) == 0:
                    print('Step {}, iteration {}, success {}/{}\n'
                          'adv_loss: {:.4f}, dist_loss: {:.4f}'.
                          format(binary_step, iteration, success_num, B,
                                 (adv_loss.item()+lfc_adv_loss.item()), dist_loss.item()))


                dist_val = torch.sqrt(torch.sum(
                    (adv_data - ori_data) ** 2, dim=[1, 2])).\
                    detach().cpu().numpy()
                pred_val = pred.detach().cpu().numpy()
                lfc_pred_val = lfc_pred.detach().cpu().numpy()
                input_val = adv_data.detach().cpu().numpy()


                for e, (dist, pred, lfc_pred, label, ii) in \
                        enumerate(zip(dist_val, pred_val, lfc_pred_val, label_val, input_val)):
                    if dist < o_bestdist[e] and pred != label and (lfc_pred != label or self.GAMMA < 0.001):
                        o_bestdist[e] = dist
                        o_bestscore[e] = pred
                        o_bestattack[e] = ii

                t4 = time.time()
                update_time += t4 - t3

                total_time += t4 - t1

                if iteration % 100 == 0:
                    print('total time: {:.2f}, for: {:.2f}, '
                          'back: {:.2f}, update: {:.2f}'.
                          format(total_time, optimize_time,
                                 clip_time, update_time))
                    total_time = 0.
                    optimize_time = 0.
                    clip_time = 0.
                    update_time = 0.
                    torch.cuda.empty_cache()

            torch.cuda.empty_cache()


        fail_idx = (o_bestscore < 0)
        o_bestattack[fail_idx] = input_val[fail_idx]

        adv_pc = torch.tensor(o_bestattack).to(adv_data)
        adv_pc = self.clip_func(adv_pc, ori_data)


        point_features = self.model.encode_pc(torch.cat((adv_data.transpose(2, 1), torch.full_like(adv_data.transpose(2, 1), 0.5)), -1))
        point_features = torch.nn.functional.normalize(point_features, dim=-1)
        logits = torch.matmul(point_features, self.text_features.T)

        if isinstance(logits, tuple):
            logits = logits[0]
        preds = torch.argmax(logits, dim=-1)

        print(preds.shape)
        success_num = (preds != target).sum().item()
        print('Successfully attack {}/{}'.format(success_num, B))
        return o_bestdist, adv_pc.detach().cpu().numpy().transpose((0, 2, 1)), success_num
