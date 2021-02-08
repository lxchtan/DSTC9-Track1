
python data_util/add_punctuation.py --dataset train --dataroot data --newDataroot data_modify/add_stop
python data_util/add_punctuation.py --dataset val --dataroot data --newDataroot data_modify/add_stop

python data_util/add_punctuation.py --dataset train --dataroot data --newDataroot data_modify/add_stop --knowledge