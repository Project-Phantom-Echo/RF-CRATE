"""Memory-mapped equivalent of the released HuPR loader; preserve file order."""
from pathlib import Path
from collections import OrderedDict
import json
import numpy as np
import torch
from torch.utils.data import Dataset

class MappedHuPR(Dataset):
    def __init__(self,config):
        self.config=config;self.cache=Path(config['dataset_path']);self.maps=OrderedDict()
        order=json.loads((self.cache/'file_order.json').read_text())
        wanted=set(config['file_indexes']);self.entries=[];self.labels=[];self.boxes=[]
        selected=[]
        for file in order:
            name=Path(file).stem;seq=int(name.split('_')[1])
            if seq not in wanted:continue
            d=self.cache/name
            receipt=json.loads((d/'complete.json').read_text());assert receipt['frames']==600
            selected.append(seq);self.labels.append(np.load(d/'joints.npy'));self.boxes.append(np.load(d/'bbox.npy'))
            self.entries.extend((name,i) for i in range(600))
        self.labels=torch.tensor(np.concatenate(self.labels));self.boxes=torch.tensor(np.concatenate(self.boxes))
        print('Mapped HuPR:',len(selected),'sequences',len(self.entries),'frames; missing requested IDs:',sorted(wanted-set(selected)),flush=True)
    def __len__(self):return len(self.entries)
    def __getitem__(self,idx):
        name,frame=self.entries[idx]
        if name not in self.maps:
            self.maps[name]=np.load(self.cache/name/'maps.npy',mmap_mode='r')
            if len(self.maps)>8:self.maps.popitem(last=False)
        self.maps.move_to_end(name)
        x=torch.from_numpy(np.array(self.maps[name][frame],copy=True))
        if self.config['format']=='cartesian':x=torch.stack((x.real,x.imag),dim=-1)
        elif self.config['format']=='polar':x=torch.stack((x.abs(),x.angle()),dim=-1)
        else:assert self.config['format']=='complex'
        return x,self.labels[idx],self.boxes[idx]
