"""
Explanation model
"""
import torch
import tqdm
import numpy as np


class RetroExpModel(torch.nn.Module):
    def __init__(self, base_model, rec_dataset, args, pools, test_users):
        super(RetroExpModel, self).__init__()
        self.base_model = base_model
        self.rec_dataset = rec_dataset
        self.args = args
        self.test_users = test_users
        self.pools = pools

        self.u_i_exp_dict = {} 
        self.user_reclist_dict = {}

    def generate_explanations(self):
        no_exp_count = 0
        test_num = len(self.test_users)
        train = self.rec_dataset.train_dict

        for u in tqdm.tqdm(self.test_users):
            seq = np.zeros([self.args.max_seq], dtype=np.int32)
            idx = self.args.max_seq - 1
            for i in reversed(train[u]):
                seq[idx] = i
                idx -= 1
                if idx == -1: break
            padding_ids = (seq == 0).nonzero()
            padding_ids = torch.LongTensor(padding_ids).to(self.args.device)
            pool = self.pools[u]
            # get rec list
            input_user = torch.Tensor(np.array([u]))
            seq = torch.LongTensor(np.array([seq])).to(self.args.device)
            pool = torch.LongTensor(pool).to(self.args.device)
            predictions = self.base_model.predict(input_user, seq, pool)
            predictions = -predictions[0]
            target_pool_ids = predictions.argsort()[:self.args.K]

            self.user_reclist_dict[u] = target_pool_ids
            for i in range(len(target_pool_ids)):
                target_pool_id = target_pool_ids[i].unsqueeze(0)
                rest_pool_ids = torch.cat((target_pool_ids[:i], target_pool_ids[i+1:]))
                
                target_pool_id, expls = self.explain(u, seq, target_pool_id, rest_pool_ids, pool, padding_ids)
                if expls is None:
                    no_exp_count += 1
                else:
                    self.u_i_exp_dict[(u, int(target_pool_id))] = [expls]
        self.evaluation(rec_k=self.args.K)
        return True

    def get_rec_list(self, user, hist_items, hist_weights):
        hist_items = torch.from_numpy(np.array(hist_items)).unsqueeze(0).to(self.device)
        hist_weights = torch.from_numpy(np.array(hist_weights)).unsqueeze(0).to(self.device)
        scores = self.base_model(None, 
            hist_items, 
            hist_weights, 
            torch.tensor([len(hist_items[0])]).to(self.device),
            pool=self.pool)
        target_ids = torch.argsort(scores, descending=True)[:self.exp_args.rec_k]
        return target_ids

    def explain(self, u, seq, target_pool_id, rest_pool_ids, pool, padding_ids):
        exp_generator = EXPGenerator(
            self.base_model,
            u,
            seq,
            target_pool_id,
            rest_pool_ids,
            pool,
            padding_ids,
            self.args
        ).to(self.args.device)

        # optimization
        optimizer = torch.optim.Adam(exp_generator.parameters(), lr=self.args.lr, weight_decay=0)
        exp_generator.train()
        lowest_l1 = np.inf
        lowest_target = np.inf
        lowest_rest = np.inf
        lowest_loss = np.inf
        lowest_thresh = np.inf
        success = False
        for step in range(self.args.step):
            score = exp_generator()
            l1, pairwise_target, pairwise_rest, loss = exp_generator.loss(score)

            # if step % 100 == 0:
            #     print(
            #         'epoch: ', step,
            #         'l1: ', l1,
            #         'target: ', pairwise_target,
            #         'rest: ', pairwise_rest,
            #         'loss', loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (l1 + pairwise_rest + pairwise_target) < lowest_thresh:
                lowest_thresh = (l1 + pairwise_rest + pairwise_target)
                optimize_delta = exp_generator.delta.detach().to('cpu').numpy()

        # if not success:
        #     expls = None
        #     exp_num = None
        #     complexity = None
        # else:
        expls = np.argwhere(optimize_delta < self.args.mask_thresh).squeeze(axis=1)
        if len(expls) == 0:
            feature = np.array([np.argmin(np.abs(optimize_delta))])
            expls = feature
        exp_num = len(expls)
        complexity = exp_num / (len(seq[0]) - len(padding_ids))
        # return exp_generator.target_id.detach().to('cpu').numpy().squeeze(), expls, exp_num, complexity
        return exp_generator.target_pool_id.detach().to('cpu').numpy().squeeze(), expls


    def evaluation(self, rec_k):
        train = self.rec_dataset.train_dict  # only use train data as sequence, no difference for explanation
        ui_fid_dict = {}
        nums = []
        complexities = []
        ious = []
        fidelity_arr = []
        for u_i, expls in self.u_i_exp_dict.items():
            user = u_i[0]
            pool = self.pools[user]
            pool = torch.LongTensor(pool).to(self.args.device)
            seq = np.zeros([self.args.max_seq], dtype=np.int32)
            idx = self.args.max_seq - 1
            for i in reversed(train[user]):
                seq[idx] = i
                idx -= 1
                if idx == -1: break
            padding_ids = (seq == 0).nonzero()[0]
            # create counterfactual sequence
            for exp in expls[0]:
                seq[exp] = 0 
            seq = torch.LongTensor(np.array([seq])).to(self.args.device)
            input_user = torch.Tensor(np.array([user]))
            target_pool_id = torch.tensor([u_i[1]]).to(self.args.device)
            target_pool_ids = self.user_reclist_dict[user]
            scores = self.base_model.predict(input_user, seq, pool)[0]

            # compute iou
            set_1 = [target_pool_id.detach().to('cpu').numpy()]
            set_2 = []
            sorted_idx = torch.argsort(scores, descending=True)
            for id in target_pool_ids:  # check for the items in the list
                rank_t = (sorted_idx==id).nonzero()
                if rank_t >= rec_k:
                    set_2.append(id.detach().to('cpu').numpy())
            inter = np.intersect1d(set_1, set_2)
            union = np.union1d(set_1, set_2)
            iou = len(inter) / len(union)
            if iou == 0:  # not success
                fidelity_arr.append(0)
                ui_fid_dict[(u_i[0], u_i[1])] = 0
            else:
                nums.append(len(expls[0]))
                complexities.append(len(expls[0]) / (self.args.max_seq - len(padding_ids)))
                ious.append(iou)
                fidelity_arr.append(1)
                ui_fid_dict[(u_i[0], u_i[1])] = 1
        print("mean num: ", np.mean(nums))
        print("mean complexity: ", np.mean(complexities))
        print("mean iou: ", np.mean(ious))
        print("control fidelity: ", np.sum(fidelity_arr) / len(fidelity_arr))
        return ui_fid_dict

class EXPGenerator(torch.nn.Module):
    def __init__(self, base_model, user, seq, target_pool_id, rest_pool_ids, pool, padding_ids, args):
        super(EXPGenerator, self).__init__()
        # print('check base model gradients!!!')
        # exit(1)
        self.base_model = base_model
        self.user = user
        self.seq = seq
        self.pool = pool
        self.padding_ids = padding_ids
        self.args = args
        self.delta = torch.nn.Parameter(
            torch.FloatTensor(self.seq.shape[-1]).uniform_(0, 1))
        self.target_pool_id = target_pool_id
        self.rest_ids = rest_pool_ids

    def get_masked_weights(self):
        clamped_delta = torch.clamp(self.delta, 0, 1)
        hist_weights_star = self.hist_weights * clamped_delta
        return hist_weights_star.unsqueeze(0)
    
    def clamp_delta(self):
        clamped_delta = torch.clamp(self.delta, 0, 1)
        clamped_delta[self.padding_ids] = 1
        return clamped_delta

    def forward(self):
        clamped_delta = self.clamp_delta()
        score = self.base_model.predict_counter(self.user, self.seq, self.pool, clamped_delta)[0]
        return score
    
    def loss(self, score):

        rest_scores = torch.cat((score[:self.target_pool_id], score[(self.target_pool_id+1):]))
        pairwise_target =  self.args.lam * torch.nn.functional.relu(self.args.alp_1 + score[self.target_pool_id[0]] - torch.sort(rest_scores, descending=True)[0][(self.args.K-1)].item())  # original

        l1 = torch.linalg.norm((1-self.delta), ord=1)
        pairwise_rest = - self.args.gam * self.args.lam * torch.sum(score[self.rest_ids]) / len(self.rest_ids)
        loss = l1 + pairwise_target + pairwise_rest
        return l1, pairwise_target, pairwise_rest, loss
