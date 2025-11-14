from monai.transforms import Transform
from monai.transforms import (
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandCropByPosNegLabeld,
    ScaleIntensityRanged,
    DeleteItemsd,
    Spacingd,
    RandAffined,
    ConcatItemsd,
    ScaleIntensityd,
    ResizeWithPadOrCropd,
    Invertd,
    AsDiscreted,
    SaveImaged,
    RandSpatialCropd,
    RandSpatialCropSamplesd,
    NormalizeIntensityd
)
from monai.networks.layers import Norm
from functools import partial
import segmentation_models_pytorch as smp
from monai.networks.nets import UNet, SegResNet, AttentionUnet, UNETR, SwinUNETR
from monai.metrics import DiceMetric
from monai.losses import DiceLoss, DiceFocalLoss, DiceCELoss
from losses import L1DFL
from monai.networks import one_hot
import torch
import matplotlib.pyplot as plt
from glob import glob 
import pandas as pd
import numpy as np
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
import os
import nibabel as nb
import sys 
config_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.append(config_dir)
from config import DATA_FOLDER, WORKING_FOLDER
#%%
def convert_to_4digits(str_num):
    if len(str_num) == 1:
        new_num = '000' + str_num
    elif len(str_num) == 2:
        new_num = '00' + str_num
    elif len(str_num) == 3:
        new_num = '0' + str_num
    else:
        new_num = str_num
    return new_num

def create_dictionary_ctptgt(ctpaths, ptpaths, gtpaths):
    data = []
    for i in range(len(gtpaths)):
        ctpath = ctpaths[i]
        ptpath = ptpaths[i]
        gtpath = gtpaths[i]
        data.append({ 'CT':ctpath, 'PT':ptpath,'GT':gtpath})
    return data

def remove_all_extensions(filename):
    while True:
        name, ext = os.path.splitext(filename)
        if ext == '':
            return name
        filename = name
#%%
def create_data_split_files():
    """Creates filepaths data for training/validation and test images and saves 
    them as `train_filepaths.csv` and `test_filepaths.csv` files under WORKING_FOLDER/data_split/; 
    all training images will be assigned a FoldID specifying which fold (out of the 5 folds) 
    the image belongs to. If the `train_filepaths.csv` and `test_filepaths.csv` already exist, 
    this function is skipped
    """
    train_filepaths = os.path.join(WORKING_FOLDER, 'data_split', 'train_filepaths.csv')
    test_filepaths = os.path.join(WORKING_FOLDER, 'data_split', 'test_filepaths.csv')
    if os.path.exists(train_filepaths) and os.path.exists(test_filepaths):
        return 
    else:
        data_split_folder = os.path.join(WORKING_FOLDER, 'data_split')
        os.makedirs(data_split_folder, exist_ok=True)
        
        imagesTr = os.path.join(DATA_FOLDER, 'imagesTr')
        labelsTr = os.path.join(DATA_FOLDER, 'labelsTr')

        ctpaths = sorted(glob(os.path.join(imagesTr, '*0000.nii.gz')))
        ptpaths = sorted(glob(os.path.join(imagesTr, '*0001.nii.gz')))
        gtpaths = sorted(glob(os.path.join(labelsTr, '*.nii.gz')))
        imageids = [remove_all_extensions(os.path.basename(path)) for path in gtpaths]

        n_folds = 5
        part_size = len(imageids) // n_folds
        remaining_elements = len(imageids) % n_folds    
        start = 0
        train_folds = []
        for i in range(n_folds):
            end = start + part_size + (1 if i < remaining_elements else 0)
            train_folds.append(imageids[start:end])
            start = end
        
        fold_sizes = [len(fold) for fold in train_folds]
        foldids = [fold_sizes[i]*[i] for i in range(len(fold_sizes))]
        foldids = [item for sublist in foldids for item in sublist]
        
        trainfolds_data = np.column_stack((imageids, foldids, ctpaths, ptpaths, gtpaths))  
        train_df = pd.DataFrame(trainfolds_data, columns=['ImageID', 'FoldID', 'CTPATH', 'PTPATH', 'GTPATH'])
        
        train_df.to_csv(train_filepaths, index=False)

        imagesTs = os.path.join(DATA_FOLDER, 'imagesTs')
        labelsTs = os.path.join(DATA_FOLDER, 'labelsTs')
        ctpaths_test = sorted(glob(os.path.join(imagesTs, '*0000.nii.gz')))
        ptpaths_test = sorted(glob(os.path.join(imagesTs, '*0001.nii.gz')))
        gtpaths_test = sorted(glob(os.path.join(labelsTs, '*.nii.gz')))
        imageids_test = [remove_all_extensions(os.path.basename(path)) for path in gtpaths_test]
        test_data = np.column_stack((imageids_test, ctpaths_test, ptpaths_test,  gtpaths_test))
        test_df = pd.DataFrame(test_data, columns=['ImageID', 'CTPATH', 'PTPATH', 'GTPATH'])
        test_df.to_csv(test_filepaths, index=False)

#%%
def get_train_valid_data_in_dict_format(fold):
    trainvalid_fpath = os.path.join(WORKING_FOLDER, 'data_split/train_filepaths.csv')
    trainvalid_df = pd.read_csv(trainvalid_fpath)
    train_df = trainvalid_df[trainvalid_df['FoldID'] != fold]
    valid_df = trainvalid_df[trainvalid_df['FoldID'] == fold]

    ctpaths_train, ptpaths_train, gtpaths_train = list(train_df['CTPATH'].values), list(train_df['PTPATH'].values), list(train_df['GTPATH'].values)
    ctpaths_valid, ptpaths_valid, gtpaths_valid = list(valid_df['CTPATH'].values), list(valid_df['PTPATH'].values), list(valid_df['GTPATH'].values)

    train_data = create_dictionary_ctptgt(ctpaths_train, ptpaths_train,  gtpaths_train)
    valid_data = create_dictionary_ctptgt(ctpaths_valid, ptpaths_valid,  gtpaths_valid)

    return train_data, valid_data

#%%
def get_test_data_in_dict_format():
    test_fpaths = os.path.join(WORKING_FOLDER, 'data_split/test_filepaths.csv')
    test_df = pd.read_csv(test_fpaths)
    ctpaths_test, ptpaths_test,  gtpaths_test = list(test_df['CTPATH'].values), list(test_df['PTPATH'].values),  list(test_df['GTPATH'].values)
    test_data = create_dictionary_ctptgt(ctpaths_test, ptpaths_test,  gtpaths_test)
    return test_data

def get_spatial_size(input_patch_size=192):
    trsz = input_patch_size
    return (trsz, trsz, trsz)

def get_spacing():
    spc = 2
    return (spc, spc, spc)


# class PrintShape(Transform):
#     def __call__(self, data):
#         for key, value in data.items():
#             if isinstance(value, torch.Tensor):  # Ensure the data item is a tensor
#                 print(f"{key}: {value.shape}")
#         return data
    
# class DebugTransform(Transform):
#     def __call__(self, data):
#         for key in data.keys():
#             print(f"{key}: {data[key].shape}")
#         return data

def get_train_transforms(input_patch_size=128):
    spatialsize = get_spatial_size(input_patch_size)
    spacing = get_spacing()
    mod_keys = ['CT', 'PT',  'GT']
    train_transforms = Compose(
    [
        LoadImaged(keys=mod_keys, image_only=False),
        EnsureChannelFirstd(keys=mod_keys, channel_dim='no_channel'),
        CropForegroundd(keys=mod_keys, source_key='CT'),
        ScaleIntensityd(keys=['CT'], minv=0, maxv=1),
        Orientationd(keys=mod_keys, axcodes="RAS"),
        Spacingd(keys=mod_keys, pixdim=spacing, mode=('bilinear', 'bilinear', 'nearest')),
        RandCropByPosNegLabeld(
            keys=mod_keys,
            label_key='GT',
            spatial_size = spatialsize,
            pos=4,
            neg=1,
            num_samples=1,
            image_key='PT',
            image_threshold=0,
            allow_smaller=True,
        ),
        ResizeWithPadOrCropd(
            keys=mod_keys,
            spatial_size=spatialsize,
            mode='constant'
        ),
        RandAffined(
            keys=mod_keys,
            mode=('bilinear', 'bilinear', 'nearest'),
            prob=0.5,
            spatial_size = spatialsize,
            translate_range=(10,10,10),
            rotate_range=(0, 0, np.pi/15),
            scale_range=(0.1, 0.1, 0.1)),
        ConcatItemsd(keys=['CT', 'PT'], name='CTPT', dim=0),
        
        DeleteItemsd(keys=['CT','PT'])
    ])

    return train_transforms

#%%
def get_valid_transforms():
    spacing = get_spacing()
    mod_keys = ['CT', 'PT', 'GT']
    valid_transforms = Compose(
    [
        LoadImaged(keys=mod_keys, image_only=False),
        EnsureChannelFirstd(keys=mod_keys),
        CropForegroundd(keys=mod_keys, source_key='CT'),
        ScaleIntensityd(keys=['CT'], minv=0, maxv=1),
        Orientationd(keys=mod_keys, axcodes="RAS"),
        Spacingd(keys=mod_keys, pixdim=spacing, mode=('bilinear', 'bilinear', 'nearest')),
        ConcatItemsd(keys=['CT', 'PT'], name='CTPT', dim=0),
        DeleteItemsd(keys=['CT', 'PT'])
    ])

    return valid_transforms


def get_post_transforms(test_transforms, save_preds_dir):
    post_transforms = Compose([
        Invertd(
            keys="Pred",
            transform=test_transforms,
            orig_keys="GT",
            meta_keys="pred_meta_dict",
            orig_meta_keys="image_meta_dict",
            meta_key_postfix="meta_dict",
            nearest_interp=False,
            to_tensor=True,
        ),
        AsDiscreted(keys="Pred", argmax=True),
        SaveImaged(keys="Pred", meta_keys="pred_meta_dict", output_dir=save_preds_dir, output_postfix="", separate_folder=False, resample=False),
    ])
    return post_transforms

def get_kernels_strides(patch_size, spacings):
    """
    This function is only used for decathlon datasets with the provided patch sizes.
    When refering this method for other tasks, please ensure that the patch size for each spatial dimension should
    be divisible by the product of all strides in the corresponding dimension.
    In addition, the minimal spatial size should have at least one dimension that has twice the size of
    the product of all strides. For patch sizes that cannot find suitable strides, an error will be raised.
    """
    sizes, spacings = patch_size, spacings
    input_size = sizes
    strides, kernels = [], []
    while True:
        spacing_ratio = [sp / min(spacings) for sp in spacings]
        stride = [2 if ratio <= 2 and size >= 8 else 1 for (ratio, size) in zip(spacing_ratio, sizes)]
        kernel = [3 if ratio <= 2 else 1 for ratio in spacing_ratio]
        if all(s == 1 for s in stride):
            break
        for idx, (i, j) in enumerate(zip(sizes, stride)):
            if i % j != 0:
                raise ValueError(
                    f"Patch size is not supported, please try to modify the size {input_size[idx]} in the spatial dimension {idx}."
                )
        sizes = [i / j for i, j in zip(sizes, stride)]
        spacings = [i * j for i, j in zip(spacings, stride)]
        kernels.append(kernel)
        strides.append(stride)

    strides.insert(0, len(spacings) * [1])
    kernels.append(len(spacings) * [3])
    return kernels, strides
#%%    
def get_model(network_name = 'unet', input_patch_size=128):
    if network_name == 'unet':
        model = UNet(
            spatial_dims=3,
            in_channels=2,
            out_channels=2,
            channels=(16, 32, 64, 128, 256, 512),
            strides=(2, 2, 2, 2, 2),
            num_res_units=2,
            norm=Norm.BATCH
        )
    elif network_name =='segresnet':
        model = SegResNet(
            spatial_dims=3,
            blocks_down=[1, 2, 2, 4],
            blocks_up=[1, 1, 1],
            init_filters=16,
            in_channels=2,
            out_channels=2,
        )
    elif network_name == 'attunet':
        model = AttentionUnet(
            spatial_dims=3,
            in_channels=2,
            out_channels=2,
            channels=(16, 32, 64, 128, 256, 512),
            strides=(2,2,2,2,2)
        )
    elif network_name == 'unetr':
        model = UNETR(
            in_channels=2,  
            out_channels=2,
            img_size=(128,128,128),        
            feature_size=8,
            hidden_size=256,              
            mlp_dim=1024,                 
            num_heads=4,                 
            proj_type="conv",
            norm_name="instance",
            res_block=True,
        )
    elif network_name == 'swinunetr':
        model = SwinUNETR(
            img_size=(96,96,96),
            in_channels=2,
            out_channels=2,
            feature_size=48,
            use_checkpoint=True
        )
    else:
        pass
    return model


#%%
def get_loss_function(*args):
    # loss_function = DiceFocalLoss(to_onehot_y=True, softmax=True, lambda_dice=1.0, lambda_focal=1.0)
    # loss_function = DiceLoss(to_onehot_y=True, softmax=True)
    # loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
    print(args[0], args[1])
    loss_function = L1DFL(gamma=args[0], epsilon=args[1])
    return loss_function

def get_optimizer(model, learning_rate=2e-4, weight_decay=1e-5):
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    return optimizer

def get_metric():
    metric = DiceMetric(include_background=False, reduction="mean")
    return metric

def get_scheduler(optimizer, max_epochs=1000):
    scheduler = CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=0)
    return scheduler

def get_validation_sliding_window_size(network_name, input_patch_size=128):
    dict_W_for_N = {
        64:64,
        96:96,
        128:128,
        160:192,
        192:192,
        224:224,
        256:256
    }
    vlsz = dict_W_for_N[input_patch_size]
    # if network_name=='swin_2d':
    #     return (vlsz, vlsz)
    
    # else:  
    return (vlsz, vlsz, vlsz)