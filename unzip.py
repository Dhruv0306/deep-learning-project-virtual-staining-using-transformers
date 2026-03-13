import os
import zipfile
import shutil

# Path to the source ZIP file containing the dataset
zip_path = r"data\data\E-Staining DermaRepo.zip" # Replace with the data zip's file
# Directory where the extracted files will be stored
out_dir = r"data\E-Staining" # Replace With the directory path you want to data to extract

# Create the output directory if it doesn't exist
os.makedirs(out_dir, exist_ok=True) 

# Open the ZIP file for reading and extract its contents
with zipfile.ZipFile(zip_path, "r") as zf:
    # Iterate through each file/directory in the ZIP archive
    for info in zf.infolist():
        # Create the full target path for the current item
        target_path = os.path.join(out_dir, info.filename)
        
        # Handle directory entries - create the directory structure
        if info.is_dir():
            os.makedirs(target_path, exist_ok=True)
            continue

        # For files, ensure all parent directories exist before extraction
        os.makedirs(os.path.dirname(target_path), exist_ok=True)

        # Extract file using streaming to handle large files efficiently
        # Uses a 10MB buffer to minimize memory usage during extraction
        with zf.open(info, "r") as src, open(target_path, "wb") as dst:
            shutil.copyfileobj(src, dst, length=1024 * 1024 * 10)  # 10 MB buffer
