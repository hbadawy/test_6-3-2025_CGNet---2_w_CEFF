
import os
import random
import shutil

#########################################################
##################### T ##############################
#########################################################
# Define the source and destination folders
source_folder1 = 'D://Datasets//SYSU-CD_dataset_JPG//test//A'
destination_folder1 = 'D://Datasets//SYSU-CD_dataset_JPG//tran_2048-val_512-small-datasets//test//A'

source_folder2 = 'D://Datasets//SYSU-CD_dataset_JPG//test//B'
destination_folder2 = 'D://Datasets//SYSU-CD_dataset_JPG//tran_2048-val_512-small-datasets//test//B'

source_folder3 = 'D://Datasets//SYSU-CD_dataset_JPG//test//label'
destination_folder3 = 'D://Datasets//SYSU-CD_dataset_JPG//tran_2048-val_512-small-datasets//test//label'

# Make sure the destination folder exists
# os.makedirs(destination_folder, exist_ok=True)

# Get the list of all images in the source folder
image_files = [f for f in os.listdir(source_folder1) if os.path.isfile(os.path.join(source_folder1, f))]



# Shuffle the images
random.shuffle(image_files)

# Select 256 images
selected_images = image_files[:512]

# Copy the selected images to the destination folder
for image in selected_images:
    source_path1 = os.path.join(source_folder1, image)
    destination_path1 = os.path.join(destination_folder1, image)
    shutil.copy(source_path1, destination_path1)

    source_path2 = os.path.join(source_folder2, image)
    destination_path2 = os.path.join(destination_folder2, image)
    shutil.copy(source_path2, destination_path2)

    source_path3 = os.path.join(source_folder3, image)
    destination_path3 = os.path.join(destination_folder3, image)
    shutil.copy(source_path3, destination_path3)

print("256 images have been copied to the destination folder.")
