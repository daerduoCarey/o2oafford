import os
import sys
from argparse import ArgumentParser
from datagen import DataGen

parser = ArgumentParser()
parser.add_argument('--env_name', type=str, help='env name')
parser.add_argument('--data_dir', type=str, help='data directory')
parser.add_argument('--data_split', type=str, help='data split')
parser.add_argument('--num_processes', type=int, default=8, help='number of CPU cores to use')
parser.add_argument('--num_epochs', type=int, default=1, help='control the data amount')
parser.add_argument('--starting_epoch', type=int, default=0, help='if you want to run this data generation across multiple machines, you can set this parameter so that multiple data folders generated on different machines have continuous trial-id for easier merging of multiple datasets')
parser.add_argument('--out_fn', type=str, default=None, help='a file that lists all valid interaction data collection [default: None, meaning data_tuple_list.txt]. Again, this is used when you want to generate data across multiple machines. You can store the filelist on different files and merge them together to get one data_tuple_list.txt')
args = parser.parse_args()
    
if args.out_fn is None:
    args.out_fn = 'data_tuple_list.txt'

#  load cat-freq
cat2freq = dict()
with open('../stats/all_cats_cnt_freq.txt', 'r') as fin:
    for l in fin.readlines():
        cat, _, freq = l.rstrip().split()
        cat2freq[cat] = int(freq)
print(cat2freq)

# load cats
cats_train_test_split = 'train' if 'train_cat' in args.data_split else 'test'
with open(os.path.join('env_%s' % args.env_name, 'stats', 'afford_cats-%s.txt' % cats_train_test_split), 'r') as fin:
    cats = [l.rstrip() for l in fin.readlines()]
print(cats)

datagen = DataGen(args.env_name, args.num_processes)

for cat in cats:
    with open('../stats/%s-%s.txt' % (cat, args.data_split), 'r') as fin:
        for l in fin.readlines():
            shape_id = l.rstrip()
            for epoch_id in range(args.starting_epoch, args.starting_epoch+args.num_epochs):
                for trial_id in range(cat2freq[cat]):
                    #print(args.data_dir, args.data_split, shape_id, epoch_id, trial_id)
                    datagen.add_one_collect_job(args.data_dir, args.data_split, shape_id, cat, epoch_id, trial_id)

datagen.start_all()

data_tuple_list = datagen.join_all()
with open(os.path.join(args.data_dir, args.out_fn), 'w') as fout:
    for item in data_tuple_list:
        fout.write(item.split('/')[-1]+'\n')

