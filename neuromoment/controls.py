"""Inference controls preserve target labels and candidate pools while disrupting EEG evidence."""
import numpy as np
from torch.utils.data import Dataset


class ShuffledEEG(Dataset):
    def __init__(self,dataset,seed=7):
        self.dataset=dataset
        if len(dataset)<2:raise ValueError('Need >=2 events for EEG shuffle')
        order=np.random.default_rng(seed).permutation(len(dataset))
        self.eeg_source_index=np.empty_like(order);self.eeg_source_index[order]=np.roll(order,1)
        self.metadata={**dataset.metadata,'input_control':'global_eeg_derangement','input_control_seed':seed}

    def __len__(self):return len(self.dataset)

    def __getitem__(self,index):
        context,_=self.dataset[int(self.eeg_source_index[index])]
        return context,self.dataset.y[index]

    def __getattr__(self,name):return getattr(self.dataset,name)
