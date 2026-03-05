import os
import pandas as pd
import numpy as np
from pathlib import Path

class DataAcquisition:  # ← Make sure this class name matches exactly
    def __init__(self, data_path='data/raw'):
        """
        Initialize Data Acquisition class
        
        Args:
            data_path (str): Path to raw data directory
        """
        self.data_path = Path(data_path)
        self.csv_path = self.data_path / 'csv'
        self.image_path = self.data_path / 'jpeg'
        
    def load_metadata(self):
        """Load all CSV metadata files"""
        try:
            # Load metadata
            meta_df = pd.read_csv(self.csv_path / 'meta.csv')
            dicom_df = pd.read_csv(self.csv_path / 'dicom_info.csv')
            
            # Load mass datasets
            mass_train = pd.read_csv(self.csv_path / 'mass_case_description_train_set.csv')
            mass_test = pd.read_csv(self.csv_path / 'mass_case_description_test_set.csv')
            
            # Load calcification datasets
            calc_train = pd.read_csv(self.csv_path / 'calc_case_description_train_set.csv')
            calc_test = pd.read_csv(self.csv_path / 'calc_case_description_test_set.csv')
            
            return {
                'meta': meta_df,
                'dicom': dicom_df,
                'mass_train': mass_train,
                'mass_test': mass_test,
                'calc_train': calc_train,
                'calc_test': calc_test
            }
            
        except FileNotFoundError as e:
            print(f"Error: {e}")
            print("Please ensure all CSV files are in the correct directory.")
            return None
    
    
    def get_image_paths(self, dicom_df):
        """Extract and categorize image paths"""
        # Filter by series description
        full_mammogram_images = dicom_df[dicom_df.SeriesDescription == 'full mammogram images'].image_path
        cropped_images = dicom_df[dicom_df.SeriesDescription == 'cropped images'].image_path
        roi_mask_images = dicom_df[dicom_df.SeriesDescription == 'ROI mask images'].image_path
        
        # Update paths to point to local directory
        full_mammogram_images = full_mammogram_images.apply(
            lambda x: str(self.image_path / Path(x).relative_to('CBIS-DDSM/jpeg'))
        )
        cropped_images = cropped_images.apply(
            lambda x: str(self.image_path / Path(x).relative_to('CBIS-DDSM/jpeg'))
        )
        roi_mask_images = roi_mask_images.apply(
            lambda x: str(self.image_path / Path(x).relative_to('CBIS-DDSM/jpeg'))
        )
        
        return {
            'full_mammogram': full_mammogram_images,
            'cropped': cropped_images,
            'roi_mask': roi_mask_images
        }
    
    def create_image_dictionaries(self, image_paths):
        """Create dictionaries for quick image lookup"""
        full_mammogram_dict = {}
        cropped_dict = {}
        roi_mask_dict = {}
        
        for path in image_paths['full_mammogram']:
            key = Path(path).parent.name
            full_mammogram_dict[key] = path
            
        for path in image_paths['cropped']:
            key = Path(path).parent.name
            cropped_dict[key] = path
            
        for path in image_paths['roi_mask']:
            key = Path(path).parent.name
            roi_mask_dict[key] = path
            
        return {
            'full_mammogram': full_mammogram_dict,
            'cropped': cropped_dict,
            'roi_mask': roi_mask_dict
        }
    
    def fix_image_paths(self, dataset, image_dicts, dataset_type='mass'):
        """
        Fix image paths in datasets
        
        Args:
            dataset: DataFrame containing image paths
            image_dicts: Dictionary of image path dictionaries
            dataset_type: 'mass' or 'calc'
        """
        dataset_fixed = dataset.copy()
        
        # Determine column indices based on dataset type
        if dataset_type == 'mass':
            full_img_col = 'image file path'
            crop_img_col = 'cropped image file path'
            roi_img_col = 'ROI mask file path'
        else:  # calc
            full_img_col = 'image file path'
            crop_img_col = 'cropped image file path'
            roi_img_col = 'ROI mask file path'
        
        # Fix full mammogram paths
        for i, img_path in enumerate(dataset_fixed[full_img_col]):
            img_name = Path(img_path).parts[-2]  # Get folder name
            if img_name in image_dicts['full_mammogram']:
                dataset_fixed.iloc[i, dataset_fixed.columns.get_loc(full_img_col)] = \
                    image_dicts['full_mammogram'][img_name]
        
        # Fix cropped image paths
        for i, img_path in enumerate(dataset_fixed[crop_img_col]):
            img_name = Path(img_path).parts[-2]
            if img_name in image_dicts['cropped']:
                dataset_fixed.iloc[i, dataset_fixed.columns.get_loc(crop_img_col)] = \
                    image_dicts['cropped'][img_name]
        
        # Fix ROI mask paths
        for i, img_path in enumerate(dataset_fixed[roi_img_col]):
            img_name = Path(img_path).parts[-2]
            if img_name in image_dicts['roi_mask']:
                dataset_fixed.iloc[i, dataset_fixed.columns.get_loc(roi_img_col)] = \
                    image_dicts['roi_mask'][img_name]
        
        return dataset_fixed