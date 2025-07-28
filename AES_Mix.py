import torch
from torch.utils.data import Dataset
import numpy as np
from tqdm import tqdm
import torchvision.transforms as transforms
import random
import pandas as pd


def oversample_data(df, rda_ratio=1.5, lpd_ratio=1.3):
    rda_samples = df[df['label'].isin(['lrda', 'grda'])]
    lpd_samples = df[df['label'] == 'lpd']
    
    num_rda_original = len(rda_samples)
    num_rda_oversample = int(num_rda_original * (rda_ratio - 1))
    
    num_lpd_original = len(lpd_samples)
    num_lpd_oversample = int(num_lpd_original * (lpd_ratio - 1))
    
    rda_oversampled = rda_samples.sample(n=num_rda_oversample, replace=True, random_state=42)
    lpd_oversampled = lpd_samples.sample(n=num_lpd_oversample, replace=True, random_state=42)
    

    df_oversampled = pd.concat([df, rda_oversampled, lpd_oversampled]).reset_index(drop=True)
    
    print(f"Number of RDA samples: Original {num_rda_original} → After Oversampling {len(df_oversampled[df_oversampled['label'].isin(['lrda', 'grda'])])}")
    print(f"Number of LPD samples: Origina {num_lpd_original} → After Oversampling {len(df_oversampled[df_oversampled['label'] == 'lpd'])}")
    
    return df_oversampled

# def oversample_rda(df, oversample_ratio=1.5):
#     rda_samples = df[df['label'].isin(['lrda', 'grda'])]

#     num_original = len(rda_samples)
#     num_oversample = int(num_original * (oversample_ratio - 1))

#     rda_oversampled = rda_samples.sample(n=num_oversample, replace=True, random_state=42)
#     df_oversampled = pd.concat([df, rda_oversampled]).reset_index(drop=True)
    
#     print(f"Number of RDA samples: Original {num_original} → After Oversampling {len(df_oversampled[df_oversampled['label'].isin(['lrda', 'grda'])])}")
    
#     return df_oversampled

class AES_Mix(Dataset):
    def __init__(self, df, label_columns, transform=None, mixup_prob=0.3, mixup_alpha=0.2, is_train=False): 
        if is_train:
            self.df = oversample_data(df)
            
        else:
            self.df = df
        self.transform = transform
        self.label_columns = label_columns
        self.mixup_prob = mixup_prob
        self.mixup_alpha = mixup_alpha
        self.is_train = is_train
        
        self.other_indices = self.df[self.df['label'] == 'other'].index.tolist()
        self.seizure_indices = self.df[self.df['label'] == 'seizure'].index.tolist()
        
        self.rda_indices = self.df[self.df['label'].isin(['lrda', 'grda'])].index.tolist()
        self.pd_indices = self.df[self.df['label'].isin(['lpd', 'gpd'])].index.tolist()

    
    def __len__(self):
        return len(self.df)
    
    def load_and_process_data(self, path, flip=False):
        x = np.load(path, allow_pickle=True)
        x = np.nan_to_num(x, nan=-1)
        
        x_LL = x.T[0].T
        x_LP = x.T[1].T
        x_RP = x.T[2].T
        x_RL = x.T[3].T

        if flip:
            x_LL = np.fliplr(x_LL)
            x_LP = np.fliplr(x_LP)
            x_RP = np.fliplr(x_RP)
            x_RL = np.fliplr(x_RL)
        
        return x_LL, x_LP, x_RP, x_RL
    
    def mixup_data(self, channels1, channels2, alpha=0.2):
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1

        x_LL1, x_LP1, x_RP1, x_RL1 = channels1
        x_LL2, x_LP2, x_RP2, x_RL2 = channels2
        
        mixed_LL = lam * x_LL1 + (1 - lam) * x_LL2
        mixed_LP = lam * x_LP1 + (1 - lam) * x_LP2
        mixed_RP = lam * x_RP1 + (1 - lam) * x_RP2
        mixed_RL = lam * x_RL1 + (1 - lam) * x_RL2

        return mixed_LL, mixed_LP, mixed_RP, mixed_RL, lam
    
    def mixup_at_position(self, channels1, channels2):
        x_LL1, x_LP1, x_RP1, x_RL1 = channels1
        x_LL2, x_LP2, x_RP2, x_RL2 = channels2
        
        mixed_LL = x_LL1.copy()
        mixed_LP = x_LP1.copy()
        mixed_RP = x_RP1.copy()
        mixed_RL = x_RL1.copy()
        
        mixed_LL[0:21, :] = x_LL2[0:21, :]
        mixed_LP[0:21, :] = x_LP2[0:21, :]
        mixed_RP[0:21, :] = x_RP2[0:21, :]
        mixed_RL[0:21, :] = x_RL2[0:21, :]

        return mixed_LL, mixed_LP, mixed_RP, mixed_RL, 41/mixed_LL.shape[0]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        current_label = row['label']
        if current_label in ['lrda', 'grda']:
            if random.random() < 0.5:
                channels = self.load_and_process_data(row.path_4, flip=True)
            else:
                channels = self.load_and_process_data(row.path_4)
        elif current_label in ['lpd']:
            if random.random() < 0.7:
                channels = self.load_and_process_data(row.path_4, flip=True)
            else:
                channels = self.load_and_process_data(row.path_4)
        else:
            channels = self.load_and_process_data(row.path_4)
        x_LL, x_LP, x_RP, x_RL = channels
        x = np.stack((x_LL, x_LP, x_RP, x_RL), axis=0)
        x = np.expand_dims(x, axis=1)
        x = torch.Tensor(x)

        y_soft = torch.Tensor(np.array(row.filter(like='_vote').values, 'float32'))
        y_hard = torch.argmax(y_soft).item()
        
        if not self.is_train or random.random() >= self.mixup_prob:
            return self._return_original(x, y_soft, y_hard)
        
        prob = random.random()
        if current_label in ['other', 'seizure']: 
            is_RDA = False
            if prob < self.mixup_prob: 
                mix_idxes = self.pd_indices + self.other_indices + self.seizure_indices
                mix_idx = random.choice(mix_idxes)
                mix_row = self.df.iloc[mix_idx]
                other_channels = self.load_and_process_data(mix_row.path_4)

                mixed_LL, mixed_LP, mixed_RP, mixed_RL, lam = self.mixup_data(channels, other_channels, self.mixup_alpha)

                mixed_x = np.stack((mixed_LL, mixed_LP, mixed_RP, mixed_RL), axis=0)
                mixed_x = np.expand_dims(mixed_x, axis=1)
                mixed_x = torch.Tensor(mixed_x)

            else: 
                return self._return_original(x, y_soft, y_hard)
        
        elif current_label in ['lrda', 'grda']: 
            is_RDA = True
            if prob < self.mixup_prob:
                mix_idxes = self.other_indices + self.seizure_indices
                mix_idx = random.choice(mix_idxes)
                mix_row = self.df.iloc[mix_idx]
                other_channels = self.load_and_process_data(mix_row.path_4)
                mixed_LL, mixed_LP, mixed_RP, mixed_RL, lam = self.mixup_at_position(channels, other_channels)

                mixed_x = np.stack((mixed_LL, mixed_LP, mixed_RP, mixed_RL), axis=0)
                mixed_x = np.expand_dims(mixed_x, axis=1)
                mixed_x = torch.Tensor(mixed_x)

                return {
                    'x' : mixed_x,
                    'y_soft': y_soft,
                    'y_hard' : y_hard,
                    'original_data': x,
                    'original_y_soft': y_soft,
                    'original_y_hard': y_hard,
                    'is_mixup': True,
                    'lam': lam,
                    'is_RDA' : is_RDA
                }
            else:
                return self._return_original(x, y_soft, y_hard)
            
        elif current_label in ['lpd', 'gpd']: 
            is_RDA = False
            if prob < self.mixup_prob:
                mix_idxes = self.other_indices + self.seizure_indices
                mix_idx = random.choice(mix_idxes)
                mix_row = self.df.iloc[mix_idx]
                other_channels = self.load_and_process_data(mix_row.path_4)

                mixed_LL, mixed_LP, mixed_RP, mixed_RL, lam = self.mixup_data(channels, other_channels, self.mixup_alpha)

                mixed_x = np.stack((mixed_LL, mixed_LP, mixed_RP, mixed_RL), axis=0)
                mixed_x = np.expand_dims(mixed_x, axis=1)
                mixed_x = torch.Tensor(mixed_x)
                
            else:
                return self._return_original(x, y_soft, y_hard)
            
        
        y_soft_2 = torch.Tensor(np.array(mix_row.filter(like='_vote').values, 'float32'))
        y_hard_2 = torch.argmax(y_soft_2).item()

        mixup_soft = lam * y_soft + (1 - lam) * y_soft_2
        mixup_hard = torch.argmax(mixup_soft).item()

        return {
            'x' : mixed_x,
            'y_soft' : mixup_soft,
            'y_hard' : mixup_hard,
            'original_data' : x,
            'original_y_soft': y_soft,
            'original_y_hard': y_hard,
            'mix_y_soft': mixup_soft,
            'mix_y_hard': mixup_hard,
            'is_mixup': True,
            'lam': lam,
            'is_RDA' : is_RDA
        }
    
    def _return_original(self, x, y_soft, y_hard):
        return {
            'x': x,
            'y_soft': y_soft,
            'y_hard': y_hard,
            'original_data': x,
            'original_y_soft': y_soft,
            'original_y_hard': y_hard,
            'mix_y_soft': y_soft, 
            'mix_y_hard': y_hard,
            'is_mixup': False,
            'lam': 1.0,
            'is_RDA' : False
        }

def AES_Mix_collate_fn(batch):
    return {
        'x': torch.stack([d['x'] for d in batch]),
        'y_soft': torch.stack([d['y_soft'] for d in batch]),
        'y_hard': torch.tensor([d['y_hard'] for d in batch]),
        'is_mixup': torch.tensor([d['is_mixup'] for d in batch]),
        'lam': torch.tensor([d['lam'] for d in batch]),
        'original_data': torch.stack([d['original_data'] for d in batch]),
        'original_y_soft': torch.stack([d['original_y_soft'] for d in batch]),
        'original_y_hard': torch.tensor([d['original_y_hard'] for d in batch]),
        'is_RDA' : torch.tensor([d['is_RDA'] for d in batch])
    }
