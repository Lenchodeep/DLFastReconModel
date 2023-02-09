
import glob
import random
import os
import shutil



def move_split(new_data_dir, source_data_list):
    os.makedirs(new_data_dir, exist_ok=True)
    for source_data in source_data_list:
        shutil.move(src=source_data, dst=new_data_dir)


def splitFile(save_path, Train_percent = 0.4, Val_percent=0.16, Test_percent=0.24):
    h5_list = glob.glob(save_path + '*.h5')
    num_files = len(h5_list)
    num_test = int(num_files*Test_percent)
    num_val = int(num_files*Val_percent)
    num_train = int(num_files * Train_percent)

    random.shuffle(h5_list)
    test_list = h5_list[:num_test]
    val_list = h5_list[num_test:(num_test+num_val)]
    train_list = h5_list[(num_test+num_val):(num_test+num_val+num_train)]
    unuse_list = h5_list[(num_test+num_val+num_train):]

    with open(save_path+'split.txt','w') as f:
        f.writelines(['train:\n'])
        [f.writelines(os.path.split(t)[1] + '\n') for t in train_list]
        f.writelines(['\nval:\n'])

        [f.writelines(os.path.split(t)[1] + '\n') for t in val_list]
        f.writelines(['\ntest:\n'])

        [f.writelines(os.path.split(t)[1] + '\n') for t in test_list]
        f.writelines(['unuse:\n'])
        [f.writelines(os.path.split(t)[1] + '\n') for t in unuse_list]
    move_split(save_path + 'train', train_list)
    move_split(save_path + 'val', val_list)
    move_split(save_path + 'test', test_list)
    move_split(save_path + 'unuse', unuse_list)


if __name__ =='__main__':
    path = '/home/d1/share/DLreconstruction/Data/FastMRIT1/knee/'
    splitFile(path)