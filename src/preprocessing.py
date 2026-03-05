import pandas as pd
import numpy as np

class DataPreprocessor:
    def __init__(self):
        pass
    
    def clean_mass_data(self, mass_df):
        """Clean and preprocess mass dataset"""
        df = mass_df.copy()
        
        # Rename columns
        df = df.rename(columns={
            'left or right breast': 'left_or_right_breast',
            'image view': 'image_view',
            'abnormality id': 'abnormality_id',
            'abnormality type': 'abnormality_type',
            'mass shape': 'mass_shape',
            'mass margins': 'mass_margins',
            'image file path': 'image_file_path',
            'cropped image file path': 'cropped_image_file_path',
            'ROI mask file path': 'ROI_mask_file_path'
        })
        
        # Handle missing values
        df['mass_shape'] = df['mass_shape'].fillna(method='bfill')
        df['mass_margins'] = df['mass_margins'].fillna(method='bfill')
        
        return df
    
    def clean_calc_data(self, calc_df):
        """Clean and preprocess calcification dataset"""
        df = calc_df.copy()
        
        # Rename columns
        df = df.rename(columns={
            'breast density': 'breast_density',
            'left or right breast': 'left_or_right_breast',
            'image view': 'image_view',
            'abnormality id': 'abnormality_id',
            'abnormality type': 'abnormality_type',
            'calc type': 'calc_type',
            'calc distribution': 'calc_distribution',
            'image file path': 'image_file_path',
            'cropped image file path': 'cropped_image_file_path',
            'ROI mask file path': 'ROI_mask_file_path'
        })
        
        # Handle missing values
        df['calc_type'] = df['calc_type'].fillna(method='bfill')
        df['calc_distribution'] = df['calc_distribution'].fillna(method='bfill')
        
        return df