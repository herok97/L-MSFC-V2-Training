NUM_PROCESSES=$1
mkdir -p ./openImages/train ./openImages/val
python downloader.py oiv_val_split.txt --num_processes ${NUM_PROCESSES} --download_folder ./openImages/val
python downloader.py oiv_train_split.txt --num_processes ${NUM_PROCESSES} --download_folder ./openImages/train
