import os
import pickle
from tqdm import tqdm

target_file = '/home/etri/DATASET/nuscenes/nuscenes_map_infos_temporal_val.pkl'
target_dir = '/home/etri/DATASET/nuscenes/OnlineHDmapV2/valid'

os.makedirs(target_dir, exist_ok=True)

with open(target_file, 'rb') as f:
    records = pickle.load(f)

for record in tqdm(records['infos'], desc='Saving pickles'):
    token = record['token']
    out_path = os.path.join(target_dir, f'{token}.pkl')
    with open(out_path, 'wb') as f:
        pickle.dump(record, f)

print(f'Saved {len(records)} pickle files to {target_dir}')
