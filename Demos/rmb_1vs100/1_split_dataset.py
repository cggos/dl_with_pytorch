import os
import random
import shutil

if __name__ == '__main__':
    random.seed(1)

    dataset_dir = os.path.join("/dev_sdb/datasets", "rmb_1vs100")  # where does dataset come from?

    split_dir = os.path.join("/tmp", "rmb_split")  # where does dataset be splitted to?
    train_dir = os.path.join(split_dir, "train")
    valid_dir = os.path.join(split_dir, "valid")
    test_dir = os.path.join(split_dir, "test")

    train_pct = 0.8
    valid_pct = 0.1
    test_pct = 0.1

    for root, dirs, files in os.walk(dataset_dir):
        for sub_dir in dirs:

            imgs = os.listdir(os.path.join(root, sub_dir))
            imgs = list(filter(lambda x: x.endswith('.jpg'), imgs))
            random.shuffle(imgs)
            img_count = len(imgs)

            train_point = int(img_count * train_pct)
            valid_point = int(img_count * (train_pct + valid_pct))

            for i in range(img_count):
                if i < train_point:
                    out_dir = os.path.join(train_dir, sub_dir)
                elif i < valid_point:
                    out_dir = os.path.join(valid_dir, sub_dir)
                else:
                    out_dir = os.path.join(test_dir, sub_dir)

                if not os.path.exists(out_dir):
                    os.makedirs(out_dir)

                target_path = os.path.join(out_dir, imgs[i])
                src_path = os.path.join(dataset_dir, sub_dir, imgs[i])

                shutil.copy(src_path, target_path)

            print('Class:{}, train:{}, valid:{}, test:{}'.format(
                sub_dir, train_point, valid_point - train_point, img_count - valid_point))
