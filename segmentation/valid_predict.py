#%%
from monai.transforms import (
    AsDiscrete,
    Compose,
)
import nibabel as nb
import argparse
from monai.inferers import sliding_window_inference
from monai.data import CacheDataset, DataLoader, Dataset,decollate_batch
import torch
import matplotlib.pyplot as plt
import os
import pandas as pd
import time
from monai.transforms import (
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandCropByPosNegLabeld,
    DeleteItemsd,
    Spacingd,
    RandAffined,
    ConcatItemsd,
    ScaleIntensityd,
    ScaleIntensityRanged,
    ResizeWithPadOrCropd,
    Invertd,
    AsDiscreted,
    SaveImaged
)
import torch
import matplotlib.pyplot as plt
from glob import glob 
import pandas as pd
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR
import os
import sys 
from monai.networks.layers import Norm
from monai.networks.nets import UNet,SegResNet, AttentionUnet
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import cv2
import SimpleITK as sitk 
import numpy as np  
import cc3d
#%%
DATA_FOLDER = '/data/blobfuse/PSMA_PCA_LESIONS_SEGMENTATION/data_resampled_results/data'
PSMA_SEGMENTATION_FOLDER = '/data/blobfuse/PSMA_PCA_LESIONS_SEGMENTATION/data_resampled_results/'
WORKING_FOLDER = "/home/jhubadmin/Desktop/segmentation_research/lymphoma-segmentation-dnn/"
#%%
#THINGS TO SET
parent_pred_dir = "/data/blobfuse/PSMA_PCA_LESIONS_SEGMENTATION/data_resampled_results/\
validation_predictions/segresnet/fold1/diceloss"
pred_dir = "/data/blobfuse/PSMA_PCA_LESIONS_SEGMENTATION/data_resampled_results/\
validation_predictions/segresnet/fold1/diceloss/predictions/fold1/segresnet/segresnet_fold1_randcrop128/" 
save_fig_path = "/data/blobfuse/PSMA_PCA_LESIONS_SEGMENTATION/data_resampled_results/\
validation_predictions/segresnet/fold1/diceloss/saved_imgs"

parent_model_dir = "/data/blobfuse/PSMA_PCA_LESIONS_SEGMENTATION/data_resampled_results/\
old_segresnet_results/fold1/diceloss"
validlog_fname='/data/blobfuse/PSMA_PCA_LESIONS_SEGMENTATION/data_resampled_results/\
old_segresnet_results/fold1/diceloss/logs/fold1/segresnet/segresnet_fold1_randcrop128/validlog_gpu1.csv'

fold = 1
network = 'segresnet'

#%%
def get_3darray_from_niftipath(
    path: str,
) -> np.ndarray:
    """Get a numpy array of a Nifti image using the filepath

    Args:
        path (str): path of the Nifti file

    Returns:
        np.ndarray: 3D numpy array for the image
    """
    image = sitk.ReadImage(path)
    array = np.transpose(sitk.GetArrayFromImage(image), (2,1,0))
    return array

def create_dictionary_ctptgt(ctpaths, ptpaths, gtpaths):
    data = []
    for i in range(len(gtpaths)):
        ctpath = ctpaths[i]
        ptpath = ptpaths[i]
        gtpath = gtpaths[i]
        data.append({'CT':ctpath, 'PT':ptpath, 'GT':gtpath})
    return data
#%%
trainvalid_fpath = os.path.join(WORKING_FOLDER, 'data_split/train_filepaths.csv')
trainvalid_df = pd.read_csv(trainvalid_fpath)
train_df = trainvalid_df[trainvalid_df['FoldID'] != fold]
valid_df = trainvalid_df[trainvalid_df['FoldID'] == fold]

ctpaths_train, ptpaths_train, gtpaths_train = list(train_df['CTPATH'].values), list(train_df['PTPATH'].values), list(train_df['GTPATH'].values)
ctpaths_valid, ptpaths_valid, gtpaths_valid = list(valid_df['CTPATH'].values), list(valid_df['PTPATH'].values), list(valid_df['GTPATH'].values)

train_data = create_dictionary_ctptgt(ctpaths_train, ptpaths_train, gtpaths_train)
valid_data = create_dictionary_ctptgt(ctpaths_valid, ptpaths_valid, gtpaths_valid)

ct_files = valid_df["CTPATH"].values
pt_files = valid_df["PTPATH"].values
gt_files = valid_df["GTPATH"].values
#%%
def get_spacing():
    spc = 2
    return (spc, spc, spc)
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
    return model
def get_validation_sliding_window_size(input_patch_size=128):
    dict_W_for_N = {
        64:64,
        128:128,
        160:192,
        192:192,
        224:224,
        256:256
    }
    vlsz = dict_W_for_N[input_patch_size]
    return (vlsz, vlsz, vlsz)
#%%
validlog = pd.read_csv(validlog_fname)
best_epoch = 2*(np.argmax(validlog['Metric']) + 1)
best_metric = np.max(validlog['Metric'])
#%%
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
#%%
inputsize = 128
experiment_code = f"{network}_fold{fold}_randcrop{inputsize}"
sw_roi_size = get_validation_sliding_window_size(inputsize) # get sliding_window inference size for given input patch size

save_models_dir = os.path.join(parent_model_dir,'models')
save_models_dir = os.path.join(save_models_dir, 'fold'+str(fold), network, experiment_code)
best_model_fname = 'model_ep=' + convert_to_4digits(str(best_epoch)) +'.pth'
model_path = os.path.join(save_models_dir, best_model_fname)
device = torch.device(f"cuda:0")
model = get_model(network, input_patch_size=inputsize)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
    
# initialize the location to save predicted masks
save_preds_dir = os.path.join(parent_pred_dir, f'predictions')
save_preds_dir = os.path.join(save_preds_dir, 'fold'+str(fold), network, experiment_code)
os.makedirs(save_preds_dir, exist_ok=True)

# get test data (in dictionary format for MONAI dataloader), test_transforms and post_transforms
test_transforms = get_valid_transforms()
post_transforms = get_post_transforms(test_transforms, save_preds_dir)

# initalize PyTorch dataset and Dataloader
dataset_test = Dataset(data=valid_data, transform=test_transforms)
dataloader_test = DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=4)

model.eval()
with torch.no_grad():
    for data in dataloader_test:
        inputs = data['CTPT'].to(device)
        sw_batch_size = 2
        print(sw_batch_size)
        data['Pred'] = sliding_window_inference(inputs, sw_roi_size, sw_batch_size, model)
        data = [post_transforms(i) for i in decollate_batch(data)]
        
#%%
_files = sorted(os.listdir(pred_dir))
#%%
pred_files=[]
for i in range(len(_files)):
    pred_file = os.path.join(pred_dir,_files[i])
    pred_files.append(pred_file)
def read_nii_vol(target_filename):
    img = nib.load(target_filename)
    VOL = np.array(img.dataobj)
    return VOL
#%%
for i in range(len(_files)):
    PET_Full_filename = pt_files[i]
    CT_Full_filename = ct_files[i]
    MASK_Full_filename = gt_files[i]
    MASK_PRED_filename = pred_files[i]

    PET_VOL = read_nii_vol(PET_Full_filename)
    PET_VOL = np.float64(PET_VOL)

    CT_VOL = read_nii_vol(CT_Full_filename)
    CT_VOL = np.float64(CT_VOL)

    MASK_VOL = read_nii_vol(MASK_Full_filename)
    MASK_VOL = np.uint8(MASK_VOL)

    #Predicted
    MASK_PRED_VOL = read_nii_vol(MASK_PRED_filename)
    MASK_PRED_VOL = np.uint8(MASK_PRED_VOL)

    PET_MIP_C = np.max(PET_VOL, axis=1)
    PET_MIP_C = cv2.normalize(PET_MIP_C, None, 0, 255, cv2.NORM_MINMAX)
    PET_MIP_C = PET_MIP_C.astype(np.uint8)
    PET_MIP_C = cv2.cvtColor(PET_MIP_C, cv2.COLOR_GRAY2BGR)

    MASK_MIP_C = np.max(MASK_VOL, axis=1)
    MASK_MIP_C = cv2.cvtColor(MASK_MIP_C*255, cv2.COLOR_GRAY2BGR)

    #Predicted
    MASK_PRED_MIP_C = np.max(MASK_PRED_VOL, axis=1)
    MASK_PRED_MIP_C = cv2.cvtColor(MASK_PRED_MIP_C*255, cv2.COLOR_GRAY2BGR)

    PET_MIP_S = np.max(PET_VOL, axis=0)
    PET_MIP_S = cv2.normalize(PET_MIP_S, None, 0, 255, cv2.NORM_MINMAX)
    PET_MIP_S = PET_MIP_S.astype(np.uint8)
    PET_MIP_S = cv2.cvtColor(PET_MIP_S, cv2.COLOR_GRAY2BGR)

    MASK_MIP_S = np.max(MASK_VOL, axis=0)
    MASK_MIP_S = cv2.cvtColor(MASK_MIP_S*255, cv2.COLOR_GRAY2BGR)

    #Predicted
    MASK_PRED_MIP_S = np.max(MASK_PRED_VOL, axis=0)
    MASK_PRED_MIP_S = cv2.cvtColor(MASK_PRED_MIP_S*255, cv2.COLOR_GRAY2BGR)

    contours2, hierarchy2 = cv2.findContours(MASK_MIP_C[:,:,0], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    PET_MIP_C = cv2.drawContours(PET_MIP_C, contours2, -1, (0, 255, 0), thickness=1)

    contours1, hierarchy1 = cv2.findContours(MASK_PRED_MIP_C[:,:,0], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    PET_PRED_MIP_C = cv2.drawContours(PET_MIP_C, contours1, -1, (255, 0, 0), thickness=1)

    contours4, hierarchy4 = cv2.findContours(MASK_MIP_S[:,:,0], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    PET_MIP_S = cv2.drawContours(PET_MIP_S, contours4, -1, (0, 255, 0), thickness=1)

    contours3, hierarchy3 = cv2.findContours(MASK_PRED_MIP_S[:,:,0], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    PET_PRED_MIP_S = cv2.drawContours(PET_MIP_S, contours3, -1, (255, 0, 0), thickness=1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 12))

    im1 = axes[0].imshow(np.rot90(PET_PRED_MIP_C))
    axes[0].set_title('PET Coronal view')

    im2 = axes[1].imshow(np.rot90(PET_PRED_MIP_S))
    axes[1].set_title('PET Sagittal view')

    plt.tight_layout()
    save_path = os.path.join(save_fig_path, f"{_files[i]}.png")
    fig.savefig(save_path)
    fig.clear()
    plt.close()
    


# %%
