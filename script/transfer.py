import os
import shutil
from PIL import Image
def trans(in_dir,folder,out_dir):
    out_dir = os.path.join(out_dir, folder)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    for filename in os.listdir(in_dir):
        src=os.path.join(in_dir, filename)
        dir=os.path.join(out_dir, filename)
        shutil.copy(src, dir)
in_dir='/raid/data/image/data/train/'
out_dir='/raid/data/guozg/data_train_32/'
folders = [dir for dir in sorted(os.listdir(in_dir),reverse=True) if os.path.isdir(os.path.join(in_dir, dir))]
print(folders)
folders=folders.reverse()
for i, folder in enumerate(folders):
    ifolder=os.path.join(in_dir,folder)
    trans(in_dir=ifolder,folder=folder,out_dir=out_dir)
    if i==31:
        break
