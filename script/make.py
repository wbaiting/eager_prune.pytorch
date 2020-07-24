import os
filepath='/raid/data/image/data/train'
outdir='/raid/data/guozg/data_train_32/'
files = os.listdir(filepath)
for i,v in enumerate(files):
    out=outdir+v
    print(out)
    if not os.path.exists(out):
        os.makedirs(out)