import torch
import os
import pickle
import numpy as np
import argparse
import random
from pathlib import Path
from utils.common_functions import set_seed
from datasets.preprocessing.preprocessing import data_init
from models.models import SASRec, GRU4Rec
from models.exp_models import RetroExpModel
from utils.evaluation import generate_pool


def generate_retro_explanation(args):

    # init dataset
    out_path = os.path.join(args.save_dir, args.model, args.dataset)
    rec_dataset = data_init(args, future_num=args.future_num)  # preprocess dataset

    if args.model == "sasrec":
        base_model = SASRec(rec_dataset.user_num, rec_dataset.item_num, args).to(args.device)
    elif args.model == "gru4rec":
        base_model = GRU4Rec(rec_dataset.user_num, rec_dataset.item_num, args).to(args.device)
    else:
        raise ValueError("Err: Wrong model name.")
    if args.model_path is not None:
        base_model.load_state_dict(torch.load(args.model_path))
    else:
        raise ValueError("need to train the base model first!")
    
    base_model.eval()
    for param in base_model.parameters():
        param.requires_grad = False
    
    Path(out_path).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(out_path, 'args.txt'), 'w') as f:
        f.write('\n'.join([str(k) + ',' + str(v) for k, v in sorted(vars(args).items(), key=lambda x: x[0])]))
    
    test_pools = generate_pool(rec_dataset, args.pool_size)

    # generate test users
    if args.test_user_num is not None:
        test_user_path = os.path.join('test_users', args.dataset)
        test_user_file = os.path.join(test_user_path, 'user_' + str(args.test_user_num) + '.pickle')
        Path(test_user_path).mkdir(parents=True, exist_ok=True)  # set save directory
        if os.path.exists(test_user_file):
            test_users = np.load(test_user_file, allow_pickle=True)
        else:
            test_users = random.sample(range(1, rec_dataset.user_num + 1), args.test_user_num)
            np.array(test_users).dump(test_user_file)
    else:
        test_users = range(1, rec_dataset.user_num + 1)

    # Create optimization model
    exp_model = RetroExpModel(
        base_model=base_model,
        rec_dataset=rec_dataset,
        args=args,
        pools=test_pools,
        test_users=test_users
    )

    exp_model.generate_explanations()
    with open(os.path.join(out_path, "explanation_obj_%d.pickle" % args.K), 'wb') as outp:
        pickle.dump(exp_model, outp, pickle.HIGHEST_PROTOCOL)
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, help="Random seed")
    parser.add_argument('--dataset', type=str, help="Choose from [ml-1m, beauty, video]")
    parser.add_argument('--model', type=str, help="Choose from [sasrec, gru4rec]")
    parser.add_argument('--dropout_rate', type=float, default=0.0)
    parser.add_argument('--max_seq', type=int, default=100, help="max length of user history used for prediction")
    parser.add_argument('--hidden_size', type=int, default=100, help="hidden embedding size")
    parser.add_argument('--K', type=int, default=10, help="rec list length")
    parser.add_argument('--num_blocks', type=int, default=2, help="attention block, only for sasrec")
    parser.add_argument('--num_heads', type=int, default=1, help="attention heads, only for sasrec")
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--model_path', type=str, default=None, help="saved model path")
    parser.add_argument('--future_num', type=int, default=5)
    parser.add_argument('--save_dir', type=str, default="logs_retro/", help="save retrospective information")
    parser.add_argument("--step", dest="step", type=int, default=500, help="# of steps in optimization")
    parser.add_argument("--lr", dest="lr", type=float, default=0.01, help="learning rate in optimization")
    parser.add_argument("--lam", dest="lam", type=float, default=100, help="the hyper-param for pairwise loss")
    parser.add_argument("--gam", dest="gam", type=float, default=1, help="the hyper-param for rest items")
    parser.add_argument("--alp_1", dest="alp_1", type=float, default=0.1, help="margin value for pairwise loss R1")
    parser.add_argument("--alp_2", dest="alp_2", type=float, default=0.1, help="margin value for pairwise loss R2")
    parser.add_argument("--mask_thresh", dest="mask_thresh", type=float, default=0.5, help="threshold for choosing explanations") 
    parser.add_argument("--test_user_num", dest="test_user_num", type=int, default=100, help="the # of users to generate explanation")
    parser.add_argument("--pool_size", dest="pool_size", type=int, default=-1, help="size of ranking pool, if increase, ps will decrease.")

    args = parser.parse_args()
    print(args)
    if args.seed is not None:  # set random seed
        set_seed(args.seed)
    generate_retro_explanation(args)