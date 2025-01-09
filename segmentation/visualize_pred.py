#%%
import os
from glob import glob
import numpy as np
import nibabel as nib
import cv2
import matplotlib.pyplot as plt

DATA_FOLDER="/data/blobfuse/PSMA_PCA_LESIONS_SEGMENTATION/data_resampled_results/data"
# pred_dir="/data/blobfuse/PSMA_PCA_LESIONS_SEGMENTATION/data_resampled_results/test_predictions/segresnet_preds/L1ghdl_ce/ensemble_results"
# save_fig_path="/data/blobfuse/PSMA_PCA_LESIONS_SEGMENTATION/data_resampled_results/test_predictions/segresnet_preds/L1ghdl_ce/results_figs"


pred_dir = f"/data/blobfuse/PSMA_PCA_LESIONS_SEGMENTATION/data_resampled_results/test_predictions/\
attunet_preds/L1ghdlfocal/predictions/fold1/attunet/attunet_fold1_randcrop128"
save_fig_path = "/data/blobfuse/PSMA_PCA_LESIONS_SEGMENTATION/data_resampled_results/\
single_fold_results/attunet/fold1model/L1ghdl/results_figs"


imagesTs = os.path.join(DATA_FOLDER, 'imagesTs')
labelsTs = os.path.join(DATA_FOLDER, 'labelsTs')

ctpaths = sorted(glob(os.path.join(imagesTs, '*0000.nii.gz')))
ptpaths = sorted(glob(os.path.join(imagesTs, '*0001.nii.gz')))
gtpaths = sorted(glob(os.path.join(labelsTs, '*.nii.gz')))

_files = sorted(os.listdir(pred_dir))
pred_files=[]
for i in range(len(_files)):
    pred_file = os.path.join(pred_dir,_files[i])
    pred_files.append(pred_file)
#%%    
def read_nii_vol(target_filename):
    img = nib.load(target_filename)
    VOL = np.array(img.dataobj)
    return VOL

for i in range(len(_files)):
    PET_Full_filename = ptpaths[i]
    MASK_Full_filename = gtpaths[i]
    MASK_PRED_filename = pred_files[i]

    PET_VOL = read_nii_vol(PET_Full_filename)
    PET_VOL = np.float64(PET_VOL)

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
