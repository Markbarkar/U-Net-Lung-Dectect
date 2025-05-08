import os
import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np
import random

# 设定随机种子
random.seed(42)  # 你可以选择任何整数作为种子

def normalization(imagearray):
    maxval = imagearray.max()
    minval = imagearray.min()
    imagearray = (imagearray - minval) / (maxval - minval) * 255
    imagearray = np.round(imagearray).astype(np.uint8)
    return imagearray


def binarize_image(imagearray, threshold=1):
    binary_image = np.where(imagearray > threshold, 255, 0).astype(np.uint8)
    return binary_image


# 创建训练和测试文件夹
trainimagedir = 'LUNA16/train/images'
trainsegdir = 'LUNA16/train/segmentation'
testimagedir = 'LUNA16/test/images'
testsegdir = 'LUNA16/test/segmentation'

os.makedirs(trainimagedir, exist_ok=True)
os.makedirs(trainsegdir, exist_ok=True)
os.makedirs(testimagedir, exist_ok=True)
os.makedirs(testsegdir, exist_ok=True)

# 遍历subset0到subset9文件夹
for subset in range(10):
    subset_dir = f'data/subset{subset}'
    seg_dir = 'data/seg-lungs-LUNA16'

    # 获取所有.mhd文件的列表
    for filename in os.listdir(subset_dir):
        if filename.endswith('.mhd'):
            itkimage = sitk.ReadImage(os.path.join(subset_dir, filename))
            image = sitk.GetArrayFromImage(itkimage)

            # 获取对应的分割图像
            # seg_filename = filename.replace('.mhd', '_segmentation.mhd')
            segitkimage = sitk.ReadImage(os.path.join(seg_dir, filename))
            segimage = sitk.GetArrayFromImage(segitkimage)

            # 选择中间的切片进行归一化和二值化
            mid_slice = image.shape[0] // 2
            img = image[mid_slice, :, :]
            seg_img = binarize_image(segimage[mid_slice, :, :])

            # 以0.8的概率保存到训练集，以0.2的概率保存到测试集
            if random.random() < 0.8:
                plt.imsave(os.path.join(trainimagedir, f'{filename.strip(".mhd")}.png'), img, cmap='gray')
                plt.imsave(os.path.join(trainsegdir, f'{filename.strip(".mhd")}.png'), seg_img, cmap='gray')
            else:
                plt.imsave(os.path.join(testimagedir, f'{filename.strip(".mhd")}.png'), img, cmap='gray')
                plt.imsave(os.path.join(testsegdir, f'{filename.strip(".mhd")}.png'), seg_img, cmap='gray')



# import numpy as np
# import SimpleITK as sitk
# import matplotlib.pyplot as plt
#
# def normalization(image_array):
#     max_val = image_array.max()
#     min_val = image_array.min()
#     # 归一化
#     image_array = (image_array - min_val) / (max_val - min_val) * 255
#     image_array = np.round(image_array)
#     return image_array.astype(np.uint8)
#
# def binarize_image(image_array, threshold=1):
#     # 二值化处理：高于阈值的设置为255，低于或等于阈值的设置为0
#     binary_image = np.where(image_array > threshold, 255, 0)
#     return binary_image.astype(np.uint8)
#
# case_path = 'data/subset0/1.3.6.1.4.1.14519.5.2.1.6279.6001.126264578931778258890371755354.mhd'
# seg_path = 'data/seg-lungs-LUNA16/1.3.6.1.4.1.14519.5.2.1.6279.6001.126264578931778258890371755354.mhd'
# itkimage = sitk.ReadImage(case_path)
# seg_itkimage = sitk.ReadImage(seg_path)
#
# image = sitk.GetArrayFromImage(itkimage)  # z,y,x
# seg_image = sitk.GetArrayFromImage(seg_itkimage)
# img = normalization(image[200, :, :])
# seg_img = binarize_image(seg_image[200, :, :])
#
# # 保存图像
# plt.imsave('img.png', img, cmap='gray')
# plt.imsave('seg_img.png', seg_img, cmap='gray')

# # 创建一个subplot，并排显示原始图像和分割图像
# plt.figure(figsize=(12, 6))
#
# # 显示原始图像
# plt.subplot(1, 2, 1)
# plt.imshow(img, cmap='gray')
# plt.title('Original Image')
# plt.axis('off')  # 不显示坐标轴
#
# # 显示分割图像
# plt.subplot(1, 2, 2)
# plt.imshow(seg_img, cmap='gray')
# plt.title('Segmentation Image')
# plt.axis('off')  # 不显示坐标轴
#
# # 显示图像
# plt.show()