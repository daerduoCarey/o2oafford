mkdir ../data/offlinedata-placement-train_cat_train_shape
mkdir ../data/offlinedata-placement-train_cat_test_shape
mkdir ../data/offlinedata-placement-test_cat

python gen_offline_data.py \
    --env_name placement \
    --data_dir ../data/offlinedata-placement-train_cat_train_shape \
    --data_split train_cat_train_shape \
    --num_processes 8 \
    --num_epochs 10 \
    --starting_epoch 0 \
    --out_fn data_tuple_list.txt

python gen_offline_data.py \
    --env_name placement \
    --data_dir ../data/offlinedata-placement-train_cat_test_shape \
    --data_split train_cat_test_shape \
    --num_processes 8 \
    --num_epochs 10 \
    --starting_epoch 0 \
    --out_fn data_tuple_list.txt

python gen_offline_data.py \
    --env_name placement \
    --data_dir ../data/offlinedata-placement-test_cat \
    --data_split test_cat \
    --num_processes 8 \
    --num_epochs 10 \
    --starting_epoch 0 \
    --out_fn data_tuple_list.txt

