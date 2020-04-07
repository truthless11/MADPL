# MADPL

Codes for the paper "Multi-Agent Task-Oriented Dialog Policy Learning with Role-Aware Reward Decomposition"

Cite this paper :

```
@inproceedings{takanobu2020multi,
  title={Multi-Agent Task-Oriented Dialog Policy Learning with Role-Aware Reward Decomposition},
  author={Takanobu, Ryuichi and Liang, Runze and Huang, Minlie},
  booktitle={ACL},
  year={2020}
}
```

## Data

unzip [zip](https://drive.google.com/open?id=1S2RXrXwsajrdzyyvM0ca_BLfGdb0PBgD) under `data` directory, or simply running

```
sh fetch_data.sh
```

the pre-processed data are under `data/processed_data` directory

- data preprocessing will be automatically done if `processed_data` directory does not exists when running `main.py`

### Use

the best trained model is under `data/model_madpl` directory

```
python main.py --test True --load data/model_madpl/selected > result.txt
```

## Run

Command

```
python main.py {--[option1]=[value1] --[option2]=[value2] ... }
```

Change the corresponding options to set hyper-parameters:

```
parser.add_argument('--log_dir', type=str, default='log', help='Logging directory')
parser.add_argument('--data_dir', type=str, default='multiwoz', help='Data directory')
parser.add_argument('--save_dir', type=str, default='model_multi', help='Directory to store model')
parser.add_argument('--load', type=str, default='', help='File name to load trained model')
parser.add_argument('--pretrain', type=bool, default=False, help='Set to pretrain')
parser.add_argument('--test', type=bool, default=False, help='Set to inference')
parser.add_argument('--config', type=str, default='multiwoz', help='Dataset to use')
parser.add_argument('--test_case', type=int, default=1000, help='Number of test cases')
parser.add_argument('--save_per_epoch', type=int, default=4, help="Save model every XXX epoches")
parser.add_argument('--print_per_batch', type=int, default=200, help="Print log every XXX batches")

parser.add_argument('--epoch', type=int, default=48, help='Max number of epoch')
parser.add_argument('--process', type=int, default=8, help='Process number')
parser.add_argument('--batchsz', type=int, default=32, help='Batch size')
parser.add_argument('--batchsz_traj', type=int, default=512, help='Batch size to collect trajectories')
parser.add_argument('--policy_weight_sys', type=float, default=2.5, help='Pos weight on system policy pretraining')
parser.add_argument('--policy_weight_usr', type=float, default=4, help='Pos weight on user policy pretraining')
parser.add_argument('--lr_policy', type=float, default=1e-3, help='Learning rate of dialog policy')
parser.add_argument('--lr_vnet', type=float, default=3e-5, help='Learning rate of value network')
parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay (L2 penalty)')
parser.add_argument('--gamma', type=float, default=0.99, help='Discounted factor')
parser.add_argument('--clip', type=float, default=10, help='Gradient clipping')
parser.add_argument('--interval', type=int, default=400, help='Update interval of target network')
```

We have implemented *distributed RL* for parallel trajectory sampling. You can set `--process` to change the number of multi-process, and set `--batchsz_traj` to change the number of trajectories each process collects before one update iteration.

### pretrain

```
python main.py --pretrain True --save_dir model_pre
```

**NOTE**: please pretrain the model first

### train

```
python main.py --load model_pre/best --lr_policy 1e-4 --save_dir model_RL --save_per_epoch 1
```

### test

```
python main.py --test True --load model_RL/best
```

## Requirements

python 3

pytorch >= 1.2