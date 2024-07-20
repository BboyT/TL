import os
import random
import numpy as np
import cv2
import SimpleITK as sitk
from skimage import transform
import pandas as pd



def resize_img_keep_ratio(img_name,target_size):
    img = img_name # 读取图片
    old_size= img.shape[0:2] # 原始图像大小
    ratio = min(float(target_size[i])/(old_size[i]) for i in range(len(old_size))) # 计算原始图像宽高与目标图像大小的比例，并取其中的较小值
    new_size = tuple([int(i*ratio) for i in old_size]) # 根据上边求得的比例计算在保持比例前提下得到的图像大小
    img = cv2.resize(img,(new_size[1], new_size[0])) # 根据上边的大小进行放缩
    pad_w = target_size[1] - new_size[1] # 计算需要填充的像素数目（图像的宽这一维度上）
    pad_h = target_size[0] - new_size[0] # 计算需要填充的像素数目（图像的高这一维度上）
    top,bottom = pad_h//2, pad_h-(pad_h//2)
    left,right = pad_w//2, pad_w -(pad_w//2)
    img_new = cv2.copyMakeBorder(img,top,bottom,left,right,cv2.BORDER_CONSTANT,None,(0,0,0))
    return img_new



def generate_train_test(roots: str):
    global CTs
    assert os.path.exists(roots), "dataset root:{} does not exist.".format(roots)
    floders = os.listdir(roots)
    train_nod = []
    files_id = []

    for folder in floders:
        files = os.listdir(os.path.join(roots, folder).replace("\\", "/"))
        files_id.append(folder)
        for file in files:
            if "image.nii" in file:
                file_path = os.path.join(roots, folder, file).replace("\\", "/")
                itkimage = sitk.ReadImage(file_path)
                numpyImage = sitk.GetArrayFromImage(itkimage).astype('float32')
                CTs = np.array(numpyImage)

        for file in files:
            if "mask.nii" in file:
                file_path = os.path.join(roots, folder, file).replace("\\", "/")
                itkimage = sitk.ReadImage(file_path)
                numpyImage = sitk.GetArrayFromImage(itkimage).astype('float32')
                Masks = np.array(numpyImage)
                areas = [np.count_nonzero(slice) for slice in Masks]
                # 找到面积最大的一层的索引
                max_area_index = np.argmax(areas)
                single_ct = CTs[max_area_index]
                nods = np.stack((single_ct,) * 3, axis=-1).squeeze().astype(np.uint8)
                train_nod.append(nods)

    files_id = pd.DataFrame(files_id)
    print("train_nod:", len(train_nod))
    print("files_id", len(files_id))

    return train_nod ,files_id



if __name__ == '__main__':
    roots = "../nii/"
    train_nod ,files_id = generate_train_test(roots)

