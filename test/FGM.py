


import torch
import numpy as np


class FGM:


    def __init__(self, model, adv_func, budget,
                 dist_metric='l2', text_features=None):

        self.model = model. cuda()
        self.model.eval()
        self.text_features = text_features
        self. adv_func = adv_func
        self.budget = budget
        self. dist_metric = dist_metric. lower()

    def get_norm(self, x):

        norm = torch.sum(x ** 2, dim=[1, 2]) ** 0.5
        return norm

    def get_gradient(self, data, target, normalize=True):


        data = data. float().cuda()
        data = data.clone(). detach(). requires_grad_(True)
        target = target.long().cuda()


        if data.grad is not None:
            data.grad.zero_()


        if isinstance(self.model, torch.nn.DataParallel):
            point_features = self. model.module.encode_pc(data)
        else:
            point_features = self.model.encode_pc(data)

        point_features = torch. nn.functional.normalize(point_features, dim=-1)
        logits = torch.matmul(point_features, self.text_features. T)

        if isinstance(logits, tuple):
            logits = logits[0]

        pred = torch.argmax(logits, dim=-1)


        loss = self.adv_func(logits, target)
        loss.backward()


        if data.grad is None:
            print("❌ ERROR: Gradient is None!  Computation graph is broken.")
            return torch.zeros_like(data), pred

        grad = data.grad.detach()

        if normalize:
            norm = self.get_norm(grad)
            grad = grad / (norm[:, None, None] + 1e-9)

        return grad, pred

    def attack(self, data, target):

        data = data.float(). cuda(). detach()
        pc = data. clone().detach()
        target = target.long(). cuda()


        normalized_grad, _ = self.get_gradient(pc, target)
        perturbation = normalized_grad * self.budget

        with torch.no_grad():

            data = data - perturbation

        torch.cuda.empty_cache()
        return data. detach(). cpu().numpy(), 0


class IFGM(FGM):


    def __init__(self, model, adv_func, clip_func,
                 budget, step_size, num_iter,
                 dist_metric='l2', text_features=None):

        super(IFGM, self).__init__(model, adv_func, budget, dist_metric, text_features)
        self.clip_func = clip_func
        self. step_size = step_size
        self. num_iter = num_iter

    def attack(self, data, target):

        B, K, C = data.shape
        data = data. float().cuda().detach()


        pc = data.clone(). detach() + torch.randn_like(data). cuda() * 1e-7
        ori_pc = data.clone().detach()
        target = target.long(). cuda()


        with torch.no_grad():
            if isinstance(self.model, torch.nn. DataParallel):
                init_features = self.model. module.encode_pc(ori_pc)
            else:
                init_features = self. model.encode_pc(ori_pc)
            init_features = torch. nn.functional.normalize(init_features, dim=-1)
            init_logits = torch.matmul(init_features, self.text_features. T)
            init_pred = torch. argmax(init_logits, dim=-1)


        for iteration in range(self. num_iter):

            normalized_grad, pred = self.get_gradient(pc, target)


            success_num = (pred == target).sum().item()

            if iteration % (self.num_iter // 5) == 0:
                print('iter {}/{}, success: {}/{}'.format(
                    iteration, self.num_iter, success_num, B))
                torch.cuda.empty_cache()

            perturbation = self.step_size * normalized_grad


            with torch.no_grad():

                pc = pc - perturbation
                pc = self.clip_func(pc, ori_pc)


        with torch.no_grad():
            if isinstance(self.model, torch.nn.DataParallel):
                final_features = self.model.module. encode_pc(pc)
            else:
                final_features = self.model. encode_pc(pc)
            final_features = torch.nn. functional.normalize(final_features, dim=-1)
            final_logits = torch.matmul(final_features, self.text_features. T)
            final_pred = torch. argmax(final_logits, dim=-1)


            success_num = (final_pred == target).sum().item()
            fooled_num = (final_pred != init_pred).sum(). item()

        return pc. detach().cpu(). numpy(), success_num


class MIFGM(FGM):


    def __init__(self, model, adv_func, clip_func,
                 budget, step_size, num_iter, mu=1.,
                 dist_metric='l2', text_features=None):

        super(MIFGM, self).__init__(model, adv_func, budget, dist_metric, text_features)
        self. clip_func = clip_func
        self. step_size = step_size
        self.num_iter = num_iter
        self.mu = mu

    def attack(self, data, target):

        B, K, C = data. shape
        data = data.float().cuda().detach()
        pc = data.clone(). detach() + torch.randn_like(data). cuda() * 1e-7
        ori_pc = data.clone().detach()
        target = target.long().cuda()
        momentum_g = torch.zeros_like(data). cuda()

        for iteration in range(self.num_iter):
            grad, pred = self. get_gradient(pc, target, normalize=False)
            success_num = (pred == target).sum(). item()

            if iteration % (self.num_iter // 5) == 0:
                print('iter {}/{}, success: {}/{}'.format(
                    iteration, self. num_iter, success_num, B))
                torch.cuda.empty_cache()


            grad_l1_norm = torch.sum(torch.abs(grad), dim=[1, 2])
            normalized_grad = grad / (grad_l1_norm[:, None, None] + 1e-9)
            momentum_g = self. mu * momentum_g + normalized_grad
            g_norm = self.get_norm(momentum_g)
            normalized_g = momentum_g / (g_norm[:, None, None] + 1e-9)
            perturbation = self.step_size * normalized_g

            with torch.no_grad():
                pc = pc - perturbation
                pc = self.clip_func(pc, ori_pc)

        with torch.no_grad():
            if isinstance(self.model, torch.nn.DataParallel):
                point_features = self. model.module.encode_pc(pc)
            else:
                point_features = self.model.encode_pc(pc)
            point_features = torch.nn. functional.normalize(point_features, dim=-1)
            logits = torch.matmul(point_features, self.text_features.T)
            if isinstance(logits, tuple):
                logits = logits[0]
            pred = torch.argmax(logits, dim=-1)
            success_num = (pred == target).sum(). item()

        print('Final success: {}/{}'.format(success_num, B))
        return pc.detach().cpu().numpy(), success_num


class PGD(IFGM):


    def __init__(self, model, adv_func, clip_func,
                 budget, step_size, num_iter,
                 dist_metric='l2', text_features=None):

        super(PGD, self).__init__(model, adv_func, clip_func,
                                  budget, step_size, num_iter,
                                  dist_metric, text_features)

    def attack(self, data, target):

        B, K, C = data.shape


        epsilon = self.budget / ((K * C) ** 0.5)
        init_perturbation = torch. empty_like(data). uniform_(-epsilon, epsilon). cuda()

        with torch.no_grad():
            init_data = data + init_perturbation

        return super(PGD, self).attack(init_data, target)


class UntargetedPGD(FGM):


    def __init__(self, model, adv_func, clip_func,
                 budget, step_size, num_iter,
                 dist_metric='l2', text_features=None):
        super(UntargetedPGD, self).__init__(model, adv_func, budget, dist_metric, text_features)
        self.clip_func = clip_func
        self. step_size = step_size
        self.num_iter = num_iter

    def get_gradient_untargeted(self, data, labels, normalize=True):

        data = data.float().cuda()
        data = data.clone(). detach().requires_grad_(True)
        labels = labels.long(). cuda()

        if data.grad is not None:
            data.grad.zero_()

        if isinstance(self.model, torch.nn. DataParallel):
            point_features = self.model. module.encode_pc(data)
        else:
            point_features = self. model.encode_pc(data)

        point_features = torch.nn. functional.normalize(point_features, dim=-1)
        logits = torch.matmul(point_features, self.text_features.T)

        if isinstance(logits, tuple):
            logits = logits[0]

        pred = torch. argmax(logits, dim=-1)


        loss = self.adv_func(logits, labels)
        loss.backward()

        if data.grad is None:
            print("❌ ERROR: Gradient is None!")
            return torch. zeros_like(data), pred

        grad = data. grad.detach()

        if normalize:
            norm = self.get_norm(grad)
            grad = grad / (norm[:, None, None] + 1e-9)

        return grad, pred

    def attack(self, data, labels):

        B, K, C = data.shape
        data = data.float(). cuda().detach()


        epsilon = self. budget / ((K * C) ** 0.5)
        init_perturbation = torch.empty_like(data).uniform_(-epsilon, epsilon).cuda()
        pc = data + init_perturbation
        ori_pc = data. clone(). detach()
        labels = labels.long(). cuda()


        with torch.no_grad():
            if isinstance(self.model, torch.nn. DataParallel):
                init_features = self.model. module.encode_pc(ori_pc)
            else:
                init_features = self. model.encode_pc(ori_pc)
            init_features = torch.nn. functional.normalize(init_features, dim=-1)
            init_logits = torch.matmul(init_features, self.text_features. T)
            init_pred = torch. argmax(init_logits, dim=-1)

        for iteration in range(self.num_iter):
            normalized_grad, pred = self.get_gradient_untargeted(pc, labels)


            fooled_num = (pred != labels).sum().item()

            if iteration % (self. num_iter // 5) == 0:
                print('iter {}/{}, fooled: {}/{}'. format(
                    iteration, self.num_iter, fooled_num, B))
                torch.cuda.empty_cache()

            perturbation = self. step_size * normalized_grad

            with torch.no_grad():

                pc = pc + perturbation
                pc = self.clip_func(pc, ori_pc)


        with torch. no_grad():
            if isinstance(self.model, torch. nn.DataParallel):
                final_features = self. model.module.encode_pc(pc)
            else:
                final_features = self.model.encode_pc(pc)
            final_features = torch.nn. functional.normalize(final_features, dim=-1)
            final_logits = torch.matmul(final_features, self.text_features. T)
            final_pred = torch. argmax(final_logits, dim=-1)

            fooled_num = (final_pred != labels).sum().item()
            changed_num = (final_pred != init_pred).sum().item()

        print(f'Final: fooled {fooled_num}/{B}, changed {changed_num}/{B}')
        return pc.detach().cpu().numpy(), fooled_num
