# Deformable Convolution Network(DCN) for head-and-neck tumor segmentation with PET/CT and MRI
3D deformable convolution network(DCN) for head and neck tumor segmentation

Code for ESTRO21 hight poster : 
```End-to-end head & neck tumor auto-segmentation using CT/PET and MRI without deformable registration```

The experiments was conducted using 'nnUNet' as a training pipeline and baseline. Please install nnUNet first and copy the DCN codes from this repo to your nnUNet folder.

To run the code, please run trainer 'nnUNetTrainerV2_200_DCN'. The image modalites order should be: CT, PET, T1 and T2. 

For guides of nnUNet please check https://github.com/MIC-DKFZ/nnUNet.
