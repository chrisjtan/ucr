import argparse
import os
import time
import torch
import torch.nn.functional
import tqdm
import numpy as np
from torch.utils.data import DataLoader
from pathlib import Path
from models.models import SASRec, GRU4Rec
from datasets.preprocessing.preprocessing import data_init
# from utils_ .data_loader import SeqDataLoader
from utils.data_loader import WarpSampler
from utils.common_functions import set_seed, set_loader_seed
from utils.evaluation import evaluate_valid, evaluate
# from utils_.evaluation import get_mrr, get_recall


def train_base(args):
    if args.timestamp:
        time_stamp = time.strftime('%b-%d-%Y_%H%M', time.localtime())
        out_path = os.path.join(args.save_dir, args.model, args.dataset + '_'
                                + time_stamp)  # create log dir
    else:
        out_path = os.path.join(args.save_dir, args.model, args.dataset)
    Path(out_path).mkdir(parents=True, exist_ok=True)  # set save directory

    with open(os.path.join(out_path, 'args.txt'), 'w') as f:
        f.write('\n'.join([str(k) + ',' + str(v) for k, v in sorted(vars(args).items(), key=lambda x: x[0])]))

    rec_dataset = data_init(args)  # preprocess dataset

    # initialize data loader
    num_batch = len(rec_dataset.train_dict) // args.batch_size
    sampler = WarpSampler(rec_dataset.train_dict, rec_dataset.user_num, rec_dataset.item_num,
                          batch_size=args.batch_size, maxlen=args.max_seq, n_workers=args.num_workers, seed=args.seed)

    if args.seed is not None:  # set random seed
        set_seed(args.seed)
        # set_loader_seed(train_loader, args.seed)  # set loader seed

    # load model
    if args.model == "sasrec":
        model = SASRec(rec_dataset.user_num, rec_dataset.item_num, args).to(args.device)
    elif args.model == "gru4rec":
        model = GRU4Rec(rec_dataset.user_num, rec_dataset.item_num, args).to(args.device)
    else:
        raise ValueError("Err: Wrong model name.")

    loss_func = torch.nn.BCEWithLogitsLoss()  # torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)

    epochs = []
    losses = []
    vals = []
    tests = []
    results = []  # used for output

    for epoch in tqdm.trange(args.num_epochs):
        if epoch == 0:
            model.eval()
            valid = evaluate_valid(model, rec_dataset, args)
            test = evaluate(model, rec_dataset, args)
            epochs.append(epoch)
            vals.append(valid)
            tests.append(test)
            results.append('epoch:%d, valid (NDCG@10: %.4f, HR@10: %.4f), test (NDCG@10: %.4f, HR@10: %.4f)'
                           % (epoch, valid[0], valid[1], test[0], test[1]))
            print(results[-1])
        model.train()
        for step in range(num_batch):
            user, seq, pos_seq, neg_seq = sampler.next_batch()
            seq = torch.LongTensor(np.array(seq)).to(args.device)
            pos_seq = torch.LongTensor(np.array(pos_seq)).to(args.device)
            neg_seq = torch.LongTensor(np.array(neg_seq)).to(args.device)

            pos_logits, neg_logits = model(user, seq, pos_seq, neg_seq)
            pos_labels, neg_labels = torch.ones(pos_logits.shape, device=args.device), torch.zeros(neg_logits.shape,
                                                                                                   device=args.device)
            indices = torch.where(seq != 0)
            loss = loss_func(pos_logits[indices], pos_labels[indices])
            loss += loss_func(neg_logits[indices], neg_labels[indices])
            losses.append(loss.detach().to('cpu'))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (epoch+1) % args.test_freq == 0:  # evaluate
            print("loss in epoch {}:{}".format(epoch, losses[-1]))
            model.eval()
            test = evaluate(model, rec_dataset, args)
            valid = evaluate_valid(model, rec_dataset, args)
            epochs.append(epoch)
            vals.append(valid)
            tests.append(test)
            results.append('epoch:%d, valid (NDCG@10: %.4f, HR@10: %.4f), test (NDCG@10: %.4f, HR@10: %.4f)'
                           % (epoch, valid[0], valid[1], test[0], test[1]))
            print(results[-1])
        
        if (epoch+1) % 100 == 0:  # save models
            torch.save(model.state_dict(), os.path.join(out_path, 'model_%d.model' % (epoch + 1)))

    print('Final Eval: valid (NDCG@10: %.4f, HR@10: %.4f), test (NDCG@10: %.4f, HR@10: %.4f)'
          % (vals[-1][0], vals[-1][1], tests[-1][0], tests[-1][1]))
    # save results
    with open(os.path.join(out_path, 'results.txt'), 'w') as f:
        for r in results:
            f.write(r + '\n')
    np.array(epochs).dump(os.path.join(out_path, 'epochs.pickle'))
    np.array(tests).dump(os.path.join(out_path, 'tests.pickle'))
    np.array(vals).dump(os.path.join(out_path, 'vals.pickle'))
    np.array(losses).dump(os.path.join(out_path, 'losses.pickle'))
    # save model params
    torch.save(model.state_dict(), os.path.join(out_path, 'model.model'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, help="Random seed")
    parser.add_argument('--dataset', type=str, help="Choose from [yelp, ml-1m]")
    parser.add_argument('--model', type=str, help="Choose from [sasrec, gru4rec]")
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=3)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--max_seq', type=int, default=100, help="max length of user history used for prediction")
    parser.add_argument('--hidden_size', type=int, default=100, help="hidden embedding size")
    parser.add_argument('--num_epochs', type=int, default=1000, help="num of training epoch")
    parser.add_argument('--num_blocks', type=int, default=2, help="attention block, only for sasrec")
    parser.add_argument('--num_heads', type=int, default=1, help="attention heads, only for sasrec")
    parser.add_argument('--dropout_rate', type=float, default=0.2)
    parser.add_argument('--l2', type=float, default=0.0, help="l2 norm")
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--model_path', type=str, default=None, help="saved model path")
    parser.add_argument('--save_dir', type=str, default="logs_base/", help="save directory")
    parser.add_argument('--test_freq', type=int, default=100, help="test for each x epochs")
    parser.add_argument('--timestamp', action='store_true', help="if save with timestamp")
    args = parser.parse_args()

    print(args)
    train_base(args)