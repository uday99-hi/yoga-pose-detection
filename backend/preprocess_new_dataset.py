import tensorflow as tf
import numpy as np
import pandas as pd 
import os
import shutil
from movenet import Movenet
import wget
import csv
import tqdm 
from data import BodyPart
from sklearn.model_selection import train_test_split
import random

# Download MoveNet model if not present
if('movenet_thunder.tflite' not in os.listdir()):
    print("Downloading MoveNet model...")
    wget.download('https://tfhub.dev/google/lite-model/movenet/singlepose/thunder/tflite/float16/4?lite-format=tflite', 'movenet_thunder.tflite')
    print("\nMoveNet model downloaded!")

movenet = Movenet('movenet_thunder')

def detect(input_tensor, inference_count=3):
    movenet.detect(input_tensor.numpy(), reset_crop_region=True)
    
    for _ in range(inference_count - 1):
        detection = movenet.detect(input_tensor.numpy(), 
                                reset_crop_region=False)
    
    return detection

class Preprocessor(object):
    """Preprocess pose samples by predicting keypoints on images and saving to CSV."""
    
    def __init__(self, images_in_folder, csvs_out_path):
        self._images_in_folder = images_in_folder
        self._csvs_out_path = csvs_out_path
        self._csvs_out_folder_per_class = 'csv_per_pose'
        self._message = []
        
        if(self._csvs_out_folder_per_class not in os.listdir()):
            os.makedirs(self._csvs_out_folder_per_class)
        
        # Get list of pose classes
        self._pose_class_names = sorted(
            [n for n in os.listdir(images_in_folder) 
             if os.path.isdir(os.path.join(images_in_folder, n))]
        )
        print(f"Found {len(self._pose_class_names)} pose classes")

    def process(self, detection_threshold=0.1):
        """Preprocess the images in the given folder."""
        for pose_class_name in self._pose_class_names:
            # Paths for pose class
            images_in_folder = os.path.join(self._images_in_folder, pose_class_name)
            csv_out_path = os.path.join(self._csvs_out_folder_per_class,
                                       pose_class_name + '.csv')
            
            print(f"\nProcessing {pose_class_name}...")
            
            # Detect landmarks in each image and write to CSV
            with open(csv_out_path, 'w', newline='') as csv_out_file:
                csv_out_writer = csv.writer(csv_out_file,
                                            delimiter=',',
                                            quoting=csv.QUOTE_MINIMAL)
                
                # Get list of images (support multiple formats)
                image_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
                image_names = sorted([
                    n for n in os.listdir(images_in_folder)
                    if any(n.endswith(ext) for ext in image_extensions)
                ])
                
                valid_image_count = 0
                
                # Detect pose landmarks in each image
                for image_name in tqdm.tqdm(image_names, desc=f"  {pose_class_name}"):
                    image_path = os.path.join(images_in_folder, image_name)
                    
                    try:
                        # Try to read as JPEG first
                        try:
                            image = tf.io.read_file(image_path)
                            image = tf.io.decode_jpeg(image)
                        except:
                            # Try PNG if JPEG fails
                            image = tf.io.read_file(image_path)
                            image = tf.io.decode_png(image)
                    except Exception as e:
                        self._message.append(f'Skipped {image_path}: Invalid image - {str(e)}')
                        continue
                    
                    # Skip images that are not RGB
                    if len(image.shape) != 3 or image.shape[2] != 3:
                        self._message.append(f'Skipped {image_path}: Image is not RGB')
                        continue
                    
                    # Resize image if needed (MoveNet expects certain dimensions)
                    image = tf.image.resize_with_pad(image, 256, 256)
                    image = tf.cast(image, dtype=tf.uint8)
                    
                    person = detect(image)
                    
                    # Save landmarks if all landmarks above threshold
                    min_landmark_score = min([keypoint.score for keypoint in person.keypoints])
                    should_keep_image = min_landmark_score >= detection_threshold
                    if not should_keep_image:
                        self._message.append(f'Skipped {image_path}: Keypoints score below threshold ({min_landmark_score:.3f})')
                        continue
                    
                    valid_image_count += 1
                    
                    # Get landmarks and scale to same size as input image
                    pose_landmarks = np.array(
                        [[keypoint.coordinate.x, keypoint.coordinate.y, keypoint.score]
                         for keypoint in person.keypoints],
                        dtype=np.float32)
                    
                    # Write landmark coordinates to CSV
                    coord = pose_landmarks.flatten().astype(str).tolist()
                    csv_out_writer.writerow([image_name] + coord)
                
                print(f"  Valid images: {valid_image_count}/{len(image_names)}")
        
        if self._message:
            print(f"\nWarnings/Errors ({len(self._message)}):")
            for msg in self._message[:10]:  # Show first 10
                print(f"  {msg}")
            if len(self._message) > 10:
                print(f"  ... and {len(self._message) - 10} more")

        # Combine all per-class CSVs into a single CSV file
        all_landmarks_df = self.all_landmarks_as_dataframe()
        all_landmarks_df.to_csv(self._csvs_out_path, index=False)
        print(f"\nCombined CSV saved to: {self._csvs_out_path}")
        print(f"Total samples: {len(all_landmarks_df)}")

    def all_landmarks_as_dataframe(self):
        """Merge all CSV files for each class into a single DataFrame."""
        total_df = None
        for class_index, class_name in enumerate(self._pose_class_names):
            csv_out_path = os.path.join(self._csvs_out_folder_per_class,
                                       class_name + '.csv')
            
            if not os.path.exists(csv_out_path):
                print(f"Warning: CSV not found for {class_name}")
                continue
            
            per_class_df = pd.read_csv(csv_out_path, header=None)
            
            if len(per_class_df) == 0:
                print(f"Warning: No data in {class_name}")
                continue
            
            # Add labels
            per_class_df['class_no'] = [class_index] * len(per_class_df)
            per_class_df['class_name'] = [class_name] * len(per_class_df)
            
            # Append folder name to filename
            per_class_df[per_class_df.columns[0]] = class_name + '/' + per_class_df[per_class_df.columns[0]]
            
            if total_df is None:
                total_df = per_class_df
            else:
                total_df = pd.concat([total_df, per_class_df], axis=0)
        
        # Create header names
        list_name = [[bodypart.name + '_x', bodypart.name + '_y', 
                      bodypart.name + '_score'] for bodypart in BodyPart]
        
        header_name = []
        for columns_name in list_name:
            header_name += columns_name
        header_name = ['filename'] + header_name
        
        header_map = {total_df.columns[i]: header_name[i]
                     for i in range(len(header_name))}
        
        total_df.rename(header_map, axis=1, inplace=True)
        
        return total_df


def split_dataset(source_folder, train_ratio=0.8):
    """Split dataset into train and test folders."""
    train_folder = os.path.join('yoga_poses', 'train')
    test_folder = os.path.join('yoga_poses', 'test')
    
    # Create folders
    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(test_folder, exist_ok=True)
    
    pose_classes = [d for d in os.listdir(source_folder) 
                   if os.path.isdir(os.path.join(source_folder, d))]
    
    print(f"\nSplitting {len(pose_classes)} pose classes into train/test...")
    
    for pose_class in pose_classes:
        source_path = os.path.join(source_folder, pose_class)
        
        # Get all images
        image_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
        images = [f for f in os.listdir(source_path)
                 if any(f.endswith(ext) for ext in image_extensions)]
        
        if len(images) == 0:
            print(f"  Warning: No images found in {pose_class}")
            continue
        
        # Shuffle and split
        random.shuffle(images)
        split_idx = int(len(images) * train_ratio)
        train_images = images[:split_idx]
        test_images = images[split_idx:]
        
        # Create class folders
        train_class_folder = os.path.join(train_folder, pose_class)
        test_class_folder = os.path.join(test_folder, pose_class)
        os.makedirs(train_class_folder, exist_ok=True)
        os.makedirs(test_class_folder, exist_ok=True)
        
        # Copy images
        for img in train_images:
            src = os.path.join(source_path, img)
            dst = os.path.join(train_class_folder, img)
            shutil.copy2(src, dst)
        
        for img in test_images:
            src = os.path.join(source_path, img)
            dst = os.path.join(test_class_folder, img)
            shutil.copy2(src, dst)
        
        print(f"  {pose_class}: {len(train_images)} train, {len(test_images)} test")


if __name__ == "__main__":
    # Configuration
    SOURCE_DATASET = r"C:\Users\ASUS\Documents\yoga datasets"
    TRAIN_RATIO = 0.8  # 80% train, 20% test
    
    print("=" * 60)
    print("Yoga Pose Dataset Preprocessing")
    print("=" * 60)
    
    # Step 1: Split dataset into train/test
    print("\nStep 1: Splitting dataset into train/test...")
    split_dataset(SOURCE_DATASET, TRAIN_RATIO)
    
    # Step 2: Preprocess training data
    print("\n" + "=" * 60)
    print("Step 2: Preprocessing training data...")
    print("=" * 60)
    images_in_folder = os.path.join('yoga_poses', 'train')
    csvs_out_path = 'train_data.csv'
    train_preprocessor = Preprocessor(images_in_folder, csvs_out_path)
    train_preprocessor.process()
    
    # Step 3: Preprocess testing data
    print("\n" + "=" * 60)
    print("Step 3: Preprocessing testing data...")
    print("=" * 60)
    images_in_folder = os.path.join('yoga_poses', 'test')
    csvs_out_path = 'test_data.csv'
    test_preprocessor = Preprocessor(images_in_folder, csvs_out_path)
    test_preprocessor.process()
    
    print("\n" + "=" * 60)
    print("Preprocessing completed!")
    print("=" * 60)
    print("\nNext step: Run training.py to train the model")




