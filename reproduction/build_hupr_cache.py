"""Convert existing RF-CRATE pickles to memory maps with exact complex64 values."""
from pathlib import Path
import pickle,json,os,argparse
import numpy as np

def main():
    p=argparse.ArgumentParser();p.add_argument('--cache',type=Path,required=True);a=p.parse_args()
    os.umask(0)
    files=json.loads((a.cache/'file_order.json').read_text())
    for name in files:
        src=Path(name);dest=a.cache/src.stem;dest.mkdir(exist_ok=True)
        if (dest/'complete.json').exists():continue
        obj=pickle.load(src.open('rb'));n=len(obj['hori']);assert n==600
        temp=dest/'maps.partial.npy';arr=np.lib.format.open_memmap(temp,mode='w+',dtype='complex64',shape=(600,64,64,8,2))
        for i in range(600):
            arr[i]=np.stack([obj['hori'][i],obj['vert'][i]],axis=-1).astype(np.complex64)
        arr.flush();del arr;os.rename(temp,dest/'maps.npy')
        np.save(dest/'joints.npy',np.asarray([x['joints'] for x in obj['labels']]))
        np.save(dest/'bbox.npy',np.asarray([x['bbox'] for x in obj['labels']]))
        mapped=np.load(dest/'maps.npy',mmap_mode='r')
        for i in [0,300,599]:assert np.array_equal(mapped[i],np.stack([obj['hori'][i],obj['vert'][i]],axis=-1).astype(np.complex64))
        (dest/'complete.json').write_text(json.dumps(dict(source=name,frames=600,exact_checked=[0,300,599]))+'\n')
        print(src.stem,'complete',flush=True)
if __name__=='__main__':main()
