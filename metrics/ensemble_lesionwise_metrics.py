#%%
import SimpleITK as sitk 
import numpy as np  
import cc3d
import pandas as pd
import numpy as np
import os
from glob import glob
#%%
#THINGS TO CHANGE
pred_dir="/data/blobfuse/PSMA_PCA_LESIONS_SEGMENTATION/data_resampled_results/test_predictions/segresnet_preds/\
dicefocal/predictions/fold1/segresnet/segresnet_fold1_randcrop128"
save_path="/data/blobfuse/PSMA_PCA_LESIONS_SEGMENTATION/data_resampled_results/single_fold_results/segresnet/\
fold1model/dicefocal"

gt_files = sorted(list(pd.read_csv('./../data_split/test_filepaths.csv')['GTPATH']))
pt_files = sorted(list(pd.read_csv('./../data_split/test_filepaths.csv')['PTPATH']))
#%%
_files = sorted(os.listdir(pred_dir))
pred_files=[]
for i in range(len(_files)):
    pred_file = os.path.join(pred_dir,_files[i])
    pred_files.append(pred_file)
#%%
def get_spacing_from_niftipath(path):
    image = sitk.ReadImage(path)
    return image.GetSpacing()

def get_3darray_from_niftipath(path: str) -> np.ndarray:
    image = sitk.ReadImage(path)
    array = np.transpose(sitk.GetArrayFromImage(image), (2,1,0))
    return array

def calculate_lesionwise_suvmean_suvmax(ptarray,maskarray,marker = 'SUVmean') -> np.float64:
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

def calculate_lesionwise_tmtv(maskarray,spacing) -> np.float64:
    voxel_volume_cc = np.prod(spacing) / 1000
    labels_out, num_lesions = cc3d.connected_components(maskarray, connectivity=18, return_N=True)
    if num_lesions == 0:
        return 0.0
    else:
        _, lesion_num_voxels = np.unique(labels_out, return_counts=True)
        lesion_num_voxels = lesion_num_voxels[1:]
        lesion_mtvs = voxel_volume_cc*lesion_num_voxels
    return lesion_mtvs

def calculate_lesionwise_tlg(ptarray,maskarray,spacing) -> np.float64:
    voxel_volume_cc = np.prod(spacing)/1000 # voxel volume in cm^3
    labels_out, num_lesions = cc3d.connected_components(maskarray, connectivity=18, return_N=True)
    tlg = []
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
        tlg = np.multiply(lesion_mtvs, lesion_suvmeans)
    return tlg
#%%
def calculate_lesion_metrics(lesion, ptarray, spacing, calculate_tmtv, calculate_suvmean_suvmax, calculate_tlg):
    """Helper function to calculate all metrics for a single lesion."""
    metrics = {
        'mtv': calculate_tmtv(lesion, spacing),
        'suvmean': calculate_suvmean_suvmax(ptarray, lesion, marker='SUVmean'),
        'suvmax': calculate_suvmean_suvmax(ptarray, lesion, marker='SUVmax'),
        'tlg': calculate_tlg(ptarray, lesion, spacing)
    }
    return metrics

def match_lesions(ptarray, ground_truth_mask, predicted_mask, spacing):
    gt_labeled, gt_num = cc3d.connected_components(ground_truth_mask, connectivity=18, return_N=True)
    pred_labeled, pred_num = cc3d.connected_components(predicted_mask, connectivity=18, return_N=True)

    gt_metrics = {}
    pred_metrics = {}
    matches = []

    for i in range(1, gt_num + 1):
        gt_lesion = gt_labeled == i
        gt_metrics[i] = calculate_lesion_metrics(gt_lesion, ptarray, spacing, calculate_lesionwise_tmtv, calculate_lesionwise_suvmean_suvmax, calculate_lesionwise_tlg)

    for j in range(1, pred_num + 1):
        pred_lesion = pred_labeled == j
        pred_metrics[j] = calculate_lesion_metrics(pred_lesion, ptarray, spacing, calculate_lesionwise_tmtv, calculate_lesionwise_suvmean_suvmax, calculate_lesionwise_tlg)

    for i in range(1, gt_num + 1):
        for j in range(1, pred_num + 1):
            #Check if there is an overlap between voxels then indicate that as a match
            if np.any((gt_labeled == i) & (pred_labeled == j)): 
                matches.append((i, j))

    return gt_metrics, pred_metrics, matches

def lesionwise_dice(maskarray, predarray, g, p):
    g_out,_ = cc3d.connected_components(maskarray, connectivity=18, return_N=True) 
    p_out,_ = cc3d.connected_components(predarray, connectivity=18, return_N=True)
    
    g_mask = np.zeros_like(g_out)
    g_mask[g_out == g] = 1

    p_mask = np.zeros_like(p_out)
    p_mask[p_out == p] = 1
    intersection = p_mask[g_mask==1]
    dice = 2 * np.sum(intersection) / (np.sum(g_mask) + np.sum(p_mask))
    return dice

def calculate_patient_level_dice_score(gtarray,predarray) -> np.float64:
    dice_score = 2.0*np.sum(predarray[gtarray == 1])/(np.sum(gtarray) + np.sum(predarray))
    return dice_score
#%%
def create_and_save_dataframe(filenames, predfiles, pt_files, gt_files):
    columns = [
        'patient_filename', 'Number of gt_lesions', 'Number of pred lesions', 'Lesion ID', 'PLvl_Dice Score'
    ]
    df = pd.DataFrame(columns=columns)
    
    for index in range(len(predfiles)):
        patient_filename = filenames[index]
        pt_array = get_3darray_from_niftipath(pt_files[index])
        mask_array = get_3darray_from_niftipath(gt_files[index])
        pred_array = get_3darray_from_niftipath(predfiles[index])
        spacing = get_spacing_from_niftipath(gt_files[index])

        gt_metrics, pred_metrics, matches = match_lesions(pt_array, mask_array, pred_array, spacing)
        dice_score = calculate_patient_level_dice_score(mask_array, pred_array)
        num_lesions_gt = len(gt_metrics)
        num_lesions_pred = len(pred_metrics)

        for gt_id in gt_metrics:
            row = {
                'patient_filename': patient_filename,
                'Number of gt_lesions': num_lesions_gt,
                'Number of pred lesions': num_lesions_pred,
                'Lesion ID': gt_id,
                'PLvl_Dice Score': dice_score
                }
            # Set ground truth metrics
            row.update({f'{metric}_gt': gt_metrics[gt_id].get(metric, np.nan) for metric in ['suvmean', 'suvmax', 'mtv', 'tlg']})
            f_match = [m for m in matches if m[0]==gt_id]
            if f_match:
                f_match = iter(f_match)
                while True:
                    match = next(f_match, None)
                    if match:
                        pred_id = match[1]
                        row.update({f'{metric}_pred': pred_metrics[pred_id].get(metric, np.nan) for metric in ['suvmean', 'suvmax', 'mtv', 'tlg']})
                        row.update({'lesionwise_dice': lesionwise_dice(mask_array,pred_array, gt_id, pred_id)})
                        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
                    else:
                        break
            else: 
                row.update({f'{metric}_pred': np.nan for metric in ['suvmean', 'suvmax', 'mtv', 'tlg']})
                row.update({'lesionwise_dice': np.nan})
                df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    return df
#%%
df = create_and_save_dataframe(_files, pred_files, pt_files, gt_files)
#%%
df.to_csv(os.path.join(save_path, 'patient_lesion_metrics.csv'), index=False)

# %%
