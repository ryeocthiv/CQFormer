import os
from PIL import Image
import shutil

# train
train_name_dir = '/home/ssh685/ICCV2023/Colour-Quantisation-main/Data/eurosat-train.txt'
dataset_root = '/home/ssh685/ICCV2023/Colour-Quantisation-main/Data/EuroSAT'
dataset_new_root = '/home/ssh685/ICCV2023/Colour-Quantisation-main/Data/EuroSAT_trainvaltest'

if not os.path.exists(os.path.join(dataset_new_root,'train')):
    print('mkdirs')
    os.makedirs(os.path.join(dataset_new_root,'train'))

with open(train_name_dir,'r') as f:
    train_name_list = f.readlines()
# print(train_name_list)


train_name_list = [train_name.replace('\n','') for train_name in train_name_list]
for train_name in train_name_list:
    clazz = train_name.split('_')[0]
    if not os.path.exists(os.path.join(dataset_new_root,'train',clazz)):
        os.makedirs(os.path.join(dataset_new_root,'train',clazz))
    img_dir = os.path.join(dataset_root,clazz,train_name)
    print(img_dir)
    # img = Image.open(img_dir)
    shutil.copy(img_dir,os.path.join(dataset_new_root,'train',clazz,train_name))

# val
val_name_dir = '/home/ssh685/ICCV2023/Colour-Quantisation-main/Data/eurosat-val.txt'

if not os.path.exists(os.path.join(dataset_new_root,'val')):
    os.makedirs(os.path.join(dataset_new_root,'val'))

with open(val_name_dir,'r') as f:
    val_name_list = f.readlines()

val_name_list = [val_name.replace('\n','') for val_name in val_name_list]
for val_name in val_name_list:
    clazz = val_name.split('_')[0]
    if not os.path.exists(os.path.join(dataset_new_root,'val',clazz)):
        os.makedirs(os.path.join(dataset_new_root,'val',clazz))
    img_dir = os.path.join(dataset_root,clazz,val_name)
    print(img_dir)
    shutil.copy(img_dir,os.path.join(dataset_new_root,'val',clazz,val_name))

# test
test_name_dir = '/home/ssh685/ICCV2023/Colour-Quantisation-main/Data/eurosat-test.txt'

if not os.path.exists(os.path.join(dataset_new_root,'test')):
    os.makedirs(os.path.join(dataset_new_root,'test'))

with open(test_name_dir,'r') as f:
    test_name_list = f.readlines()

test_name_list = [test_name.replace('\n','') for test_name in test_name_list]
for test_name in test_name_list:
    clazz = test_name.split('_')[0]
    if not os.path.exists(os.path.join(dataset_new_root,'test',clazz)):
        os.makedirs(os.path.join(dataset_new_root,'test',clazz))
    img_dir = os.path.join(dataset_root,clazz,test_name)
    print(img_dir)
    shutil.copy(img_dir,os.path.join(dataset_new_root,'test',clazz,test_name))