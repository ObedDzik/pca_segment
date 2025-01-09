#%%
import SimpleITK as sitk 
import numpy as np  
import cc3d
import pandas as pd
import numpy as np
import os
from glob import glob
#%%
PSMA_SEGMENTATION_FOLDER = '/data/blobfuse/PSMA_PCA_LESIONS_SEGMENTATION/data_resampled_results/'
WORKING_FOLDER = "/home/jhubadmin/Desktop/segmentation_research/lymphoma-segmentation-dnn/"
TRAIN_IMGS = '/data/blobfuse/PSMA_PCA_LESIONS_SEGMENTATION/data_resampled_results/data/imagesTr'
TRAIN_GT = '/data/blobfuse/PSMA_PCA_LESIONS_SEGMENTATION/data_resampled_results/data/labelsTr'
TEST_IMGS = '/data/blobfuse/PSMA_PCA_LESIONS_SEGMENTATION/data_resampled_results/data/imagesTs'
TEST_GT = '/data/blobfuse/PSMA_PCA_LESIONS_SEGMENTATION/data_resampled_results/data/labelsTs'
# RESULTS_FOLDER = "/data/blobfuse/PSMA_PCA_LESIONS_SEGMENTATION/data_resampled_results/val_results_hornet_3_3_smooth"
#%%
train_ctpaths = sorted(glob(os.path.join(TRAIN_IMGS, '*0000.nii.gz')))
train_ptpaths = sorted(glob(os.path.join(TRAIN_IMGS, '*0001.nii.gz')))

test_ctpaths = sorted(glob(os.path.join(TEST_IMGS, '*0000.nii.gz')))
test_ptpaths = sorted(glob(os.path.join(TEST_IMGS, '*0001.nii.gz')))

train_labels = sorted(glob(os.path.join(TRAIN_GT, '*.nii.gz')))
test_labels = sorted(glob(os.path.join(TEST_GT, '*.nii.gz')))

all_ctpaths = train_ctpaths + test_ctpaths
all_ptpaths = train_ptpaths + test_ptpaths
all_gtpaths = train_labels + test_labels
filenames = [path[-18:-7] for path in all_gtpaths]
#%%
def get_spacing_from_niftipath(path):
    image = sitk.ReadImage(path)
    return image.GetSpacing()
def get_3darray_from_niftipath(
    path: str,
) -> np.ndarray:
    image = sitk.ReadImage(path)
    array = np.transpose(sitk.GetArrayFromImage(image), (2,1,0))
    return array

#%%
def calculate_lesionwise_suvmean_suvmax(
    ptarray: np.ndarray, 
    maskarray: np.ndarray,
    marker: str = 'SUVmean'
) -> np.float64:

    lesion_suv_means = []
    lesion_suv_max = []

    labels_out, num_lesions = cc3d.connected_components(maskarray, connectivity=18, return_N=True)
    
    for i in range(1, num_lesions+1):
        mask = np.zeros_like(labels_out)
        mask[labels_out == i] = 1
        prod = np.multiply(mask, ptarray)
        num_nonzero_voxels = len(np.nonzero(mask)[0])
        if marker == "SUVmean":
            lesion_suv_means.append(np.sum(prod)/num_nonzero_voxels)
        elif marker == "SUVmax":
            lesion_suv_max.append(np.max(prod))            

    if marker == "SUVmean":
        return lesion_suv_means
    elif marker =="SUVmax":
        return lesion_suv_max
def get_num_lesions (maskarray):
    _, num_lesions = cc3d.connected_components(maskarray, connectivity=18, return_N=True)
    return num_lesions
def calculate_lesionwise_mtv(
    maskarray: np.ndarray,
    spacing: tuple
) -> np.float64:
    voxel_volume_cc = np.prod(spacing) / 1000
    labels_out, num_lesions = cc3d.connected_components(maskarray, connectivity=18, return_N=True)
    
    if num_lesions == 0:
        return 0.0
    else:
        _, lesion_num_voxels = np.unique(labels_out, return_counts=True)
        lesion_num_voxels = lesion_num_voxels[1:]
        lesion_mtvs = voxel_volume_cc*lesion_num_voxels
    
    return lesion_mtvs

def calculate_lesionwise_lg(
    ptarray: np.ndarray,
    maskarray: np.ndarray,
    spacing: tuple
) -> np.float64:
    """Function to return the total lesion glycolysis (TLG) using a 3D PET image 
    and the corresponding 3D segmentation mask (containing 0s for background and
    1s for lesion/tumor)
    LG = SUV1*V1 + SUV2*V2 + ... + SUVn*Vn, where SUV1...SUVn are the SUVmean 
    values of lesions 1...n with volumes V1...Vn, respectively

    Args:
        ptarray (np.ndarray): numpy ndarray for 3D PET image
        maskarray (np.ndarray): numpy ndarray for 3D mask image

    Returns:
        np.float64: total lesion glycolysis in cm^3 (assuming SUV is unitless)
    """
    voxel_volume_cc = np.prod(spacing)/1000 # voxel volume in cm^3

    labels_out, num_lesions = cc3d.connected_components(maskarray, connectivity=18, return_N=True)
    lg = []
    if num_lesions == 0:
        return 0.0
    else:
        _, lesion_num_voxels = np.unique(labels_out, return_counts=True)
        lesion_num_voxels = lesion_num_voxels[1:]
        lesion_mtvs = voxel_volume_cc*lesion_num_voxels
        
        lesion_suvmeans = []
        for i in range(1, num_lesions+1):
            mask = np.zeros_like(labels_out)
            mask[labels_out == i] = 1
            prod = np.multiply(mask, ptarray)
            num_nonzero_voxels = len(np.nonzero(mask)[0])
            lesion_suvmeans.append(np.sum(prod)/num_nonzero_voxels)
        
        lg = np.multiply(lesion_mtvs, lesion_suvmeans)
    return lg
#%%
def calculate_lesion_metrics(lesion, ptarray, spacing, calculate_mtv, calculate_suvmean_suvmax, calculate_lg):
    """Helper function to calculate all metrics for a single lesion."""
    metrics = {
        'mtv': calculate_mtv(lesion, spacing),
        'suvmean': calculate_suvmean_suvmax(ptarray, lesion, marker='SUVmean'),
        'suvmax': calculate_suvmean_suvmax(ptarray, lesion, marker='SUVmax'),
        'lg': calculate_lg(ptarray, lesion, spacing)
    }
    return metrics
def get_lesions_metric_dict(ptarray, ground_truth_mask, spacing):
    gt_labeled, gt_num = cc3d.connected_components(ground_truth_mask, connectivity=18, return_N=True)
    gt_metrics = {}

    for i in range(1, gt_num + 1):
        gt_lesion = gt_labeled == i
        gt_metrics[i] = calculate_lesion_metrics(gt_lesion, ptarray, spacing,
                                                 calculate_lesionwise_mtv, 
                                                 calculate_lesionwise_suvmean_suvmax, 
                                                 calculate_lesionwise_lg)

    return gt_metrics

#%%
def create_and_save_dataframe(filenames, pt_files, gt_files):
    columns = [
        'patient_filename', 'num_lesions', 'lesion_ID','mtv','suvmean','suvmax','lg'
    ]
    df = pd.DataFrame(columns=columns)
    
    for index in range(len(pt_files)):
        patient_filename = filenames[index]
        pt_array = get_3darray_from_niftipath(pt_files[index])
        mask_array = get_3darray_from_niftipath(gt_files[index])
        spacing = get_spacing_from_niftipath(gt_files[index])

        gt_metrics = get_lesions_metric_dict(pt_array, mask_array, spacing)
        num_lesions_gt = len(gt_metrics)

        for gt_id in gt_metrics:
            row = {
                'patient_filename': patient_filename,
                'num_lesions': num_lesions_gt,
                'lesion_ID': gt_id,
                'mtv': gt_metrics[gt_id]['mtv'],
                'suvmean':gt_metrics[gt_id]['suvmean'],
                'suvmax':gt_metrics[gt_id]['suvmax'],
                'lg': gt_metrics[gt_id]["lg"]
                }
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    return df
#%%
pt_array = get_3darray_from_niftipath(all_ptpaths[10])
mask_array = get_3darray_from_niftipath(all_gtpaths[10])
spacing = get_spacing_from_niftipath(all_gtpaths[10])
lesion_suv_mean = calculate_lesionwise_suvmean_suvmax(pt_array,mask_array,'SUVmean')
lesion_suv_max = calculate_lesionwise_suvmean_suvmax(pt_array,mask_array,'SUVmax')
num_of_lesions = get_num_lesions (mask_array)
lesion_mtvs = calculate_lesionwise_mtv(mask_array,spacing)
lesion_lg = calculate_lesionwise_lg(pt_array,mask_array,spacing)
#%%
gt_metrics = get_lesions_metric_dict(pt_array, mask_array, spacing)
#%%
df = create_and_save_dataframe(filenames, all_ptpaths, all_gtpaths)
#%%
df.to_csv('psma_data_exploration.csv', index=False)
