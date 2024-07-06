import cv2
import os
import numpy as np
from tensorflow.keras.utils import to_categorical

def image_collect(path, TRAIN_EXP_RESIZE = False):
    img_height, img_width = 224, 224
    tmp = []
    for folder in sorted(os.listdir(path)):
        class_path = path + folder + "/"
        for img in os.listdir(class_path):
            img_pth = class_path + img
            img_arr = cv2.imread(img_pth)
            if(TRAIN_EXP_RESIZE):
                img_arr = cv2.cvtColor(img_arr, cv2.COLOR_BGR2GRAY)
                img_arr = cv2.resize(img_arr, (7, 7))
            else:
                img_arr = cv2.cvtColor(img_arr, cv2.COLOR_BGR2RGB)
                img_arr = cv2.resize(img_arr, (img_height, img_width))
            img_arr = img_arr/255
            tmp.append(img_arr)
    return np.array(tmp)

def prepare_data():
    batch_size = 100
    
    x_train, x_train_exp_size, x_test, x_val = [], [], [], []

    data_path = "./dataset_copied/train/"
    x_train = image_collect(data_path)
    data_path = "./dataset_copied/train/"
    x_train_exp_size = image_collect(data_path, True)
    data_path = "./dataset_copied/val/val_images/"
    x_val = image_collect(data_path)
    data_path = "./dataset_copied/test/"
    x_test = image_collect(data_path)
    
    train_size, val_size, test_size = int(len(x_train)/4), int(len(x_val)/4), int(len(x_test)/4)
    y_train = np.concatenate([np.zeros(train_size, dtype=np.float32),
                              np.ones(train_size, dtype=np.float32), 
                              np.ones(train_size, dtype=np.float32)*2,
                              np.ones(train_size, dtype=np.float32)*3])
    y_val = np.concatenate([np.zeros(val_size, dtype=np.float32),
                              np.ones(val_size, dtype=np.float32), 
                              np.ones(val_size, dtype=np.float32)*2,
                              np.ones(val_size, dtype=np.float32)*3])
    y_test = np.concatenate([np.zeros(test_size, dtype=np.float32),
                              np.ones(test_size, dtype=np.float32), 
                              np.ones(test_size, dtype=np.float32)*2,
                              np.ones(test_size, dtype=np.float32)*3])
    y_train_arr = to_categorical(y_train)
    y_val_arr = to_categorical(y_val)
    y_test_arr = to_categorical(y_test)

    train_dataset = tf.data.Dataset.from_tensor_slices((
        x_train, y_train_arr, x_train_exp_size))
    train_dataset = train_dataset.shuffle(buffer_size=100).batch(batch_size)

    val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val_arr))
    val_dataset = val_dataset.batch(batch_size)
    
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test_arr))
    test_dataset = test_dataset.batch(batch_size)
    
    return train_dataset, val_dataset, test_dataset