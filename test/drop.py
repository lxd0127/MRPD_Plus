import numpy as np
import torch
import torch.nn.functional as F

class SaliencyDrop:


    def __init__(self, model, num_drop, alpha=1.0, k=5, text_features=None):
        self.model = model.cuda()
        self.model.eval()
        self.num_drop = num_drop
        self.alpha = alpha
        self.k = k
        self.text_features = text_features

    def get_gradient(self, data, target):


        input_data = data.clone().detach().float().cuda()
        input_data.requires_grad_(True)
        target = target.long().cuda()


        point_features = self.model.encode_pc(input_data)
        point_features = F.normalize(point_features, dim=-1)


        logits = torch.matmul(point_features, self.text_features.T)
        if isinstance(logits, tuple):
            logits = logits[0]


        loss = F.cross_entropy(logits, target)
        loss.backward()

        with torch.no_grad():
            grad = input_data.grad

            if grad.shape[-1] > 3:
                grad = grad[:, :, :3]


            if np.random.rand() < 0.05:
                grad_mag = torch.norm(grad, dim=-1)
                print(f"[DEBUG] Grad Max: {grad_mag.max().item():.2e}, Mean: {grad_mag.mean().item():.2e}, Non-zero%: {(grad_mag > 1e-8).float().mean().item()*100:.1f}%")

            pred = torch.argmax(logits, dim=-1)
            success_num = (pred == target).sum().detach().cpu().item()

        return grad, success_num

    def attack(self, data, target):

        B, K_total, C = data.shape


        full_data = data.clone().float().cuda().detach()
        target = target.long().cuda().detach()


        num_rounds = int(np.ceil(self.num_drop / self.k))

        for i in range(num_rounds):
            K = full_data.shape[1]


            current_drop_num = min(self.k, self.num_drop - i * self.k)
            if current_drop_num <= 0: break


            grad, success_num = self.get_gradient(full_data, target)


            if i == 0 or i % 5 == 0 or i == num_rounds - 1:
                print(f'Iteration {i}/{num_rounds}, Acc: {success_num}/{B} ({(success_num/B)*100:.1f}%), Points: {K}/{K_total}')

            with torch.no_grad():

                xyz = full_data[:, :, :3]

                center = torch.median(xyz, dim=1, keepdim=True)[0]

                r = torch.norm(xyz - center, dim=-1)


                grad_mag = torch.norm(grad, dim=-1)
                saliency = (r ** self.alpha) * grad_mag


                num_keep = K - current_drop_num
                _, idx = torch.topk(saliency, k=num_keep, dim=1, largest=False)


                new_data_list = []
                for j in range(B):
                    new_data_list.append(full_data[j, idx[j]])
                full_data = torch.stack(new_data_list, dim=0)


        torch.cuda.empty_cache()
        return full_data.detach().cpu().numpy(), 0
