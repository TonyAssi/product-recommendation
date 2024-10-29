from datasets import load_dataset
import os
import csv
from Embed import create_dataset_embeddings

# Image folder
img_folder = './img'
repo_name = 'tonyassi/product-images'
hf_token = 'YOUR_HF_TOKEN'

# Output CSV file name
output_csv = img_folder + '/metadata.csv'

def create_metadata_csv(img_folder, output_csv):
    # List all files in the img folder and filter for image files only
    image_extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.PNG'}
    files = sorted([f for f in os.listdir(img_folder) if os.path.splitext(f)[1].lower() in image_extensions])
    
    # Create and write the metadata to a CSV file
    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['file_name', 'id'])  # Write header
        
        # Write each filename and a unique ID
        for idx, filename in enumerate(files):
            writer.writerow([filename, idx])


# Call the function to create metadata.csv
create_metadata_csv(img_folder, output_csv)

# Load dataset and push to Hugging Face Hub
dataset = load_dataset('imagefolder', data_dir=img_folder, split='train')
dataset.push_to_hub(repo_name, token=hf_token)

# Create embeddings dataset
create_dataset_embeddings(input_dataset=repo_name,output_dataset=repo_name + '-embeddings',token=hf_token)

print('Done')



