import kagglehub
import os

os.environ["KAGGLEHUB_CACHE"] = "/data/nuscenes-full"
# Download latest version
path = kagglehub.dataset_download("mitanshuchakrawarty/nuscenes")

print("Path to dataset files:", path)