import pandas as pd
import requests
from PIL import Image
import imagehash
from io import BytesIO
import os
import json
from tqdm import tqdm

HASH_DISTANCE_THRESHOLD = 10  
LOGO_FOLDER = "downloaded_logos"
RESULT_FILE = "groups_logo.json"

os.makedirs(LOGO_FOLDER, exist_ok=True)

def get_logo_url(domain):
    return f"https://logo.clearbit.com/{domain}"

def download_logo(domain):
    logo_url = get_logo_url(domain)
    try:
        response = requests.get(logo_url, timeout=10)
        if response.status_code == 200:
            img = Image.open(BytesIO(response.content)).convert("RGB")
            path = os.path.join(LOGO_FOLDER, f"{domain}.png")
            img.save(path)
            return path
    except Exception as e:
        print(f"[Error] Couldn't download logo for {domain}: {e}")
    return None

def hash_logo(image_path):
    try:
        img = Image.open(image_path).resize((128, 128))
        return imagehash.phash(img)
    except Exception as e:
        print(f"[Error] Couldn't generate hash for {image_path}: {e}")
        return None

def group_logos_by_similarity(hashes_dict):
    groups = []
    used = set()

    keys = list(hashes_dict.keys())
    for i in range(len(keys)):
        if keys[i] in used:
            continue
        group = [keys[i]]
        used.add(keys[i])
        for j in range(i + 1, len(keys)):
            if keys[j] in used:
                continue
            dist = hashes_dict[keys[i]] - hashes_dict[keys[j]]
            if dist <= HASH_DISTANCE_THRESHOLD:
                group.append(keys[j])
                used.add(keys[j])
        groups.append(group)
    return groups

def main():
    print("[1] Reading the .parquet file...")
    df = pd.read_parquet("logos.snappy.parquet")

    if "domain" not in df.columns:
        raise Exception("Column 'domain' does not exist in the file!")

    websites = df["domain"].dropna().unique()

    print("[2] Downloading logos...")
    image_paths = {}
    for site in tqdm(websites):
        path = download_logo(site)
        if path:
            image_paths[site] = path

    print("[3] Generating perceptual hashes...")
    hashes = {}
    for site, path in tqdm(image_paths.items()):
        h = hash_logo(path)
        if h:
            hashes[site] = h

    print("[4] Grouping similar logos...")
    groups = group_logos_by_similarity(hashes)

    print(f"[5] Saving {len(groups)} groups to {RESULT_FILE}...")
    with open(RESULT_FILE, "w") as f:
        json.dump(groups, f, indent=4)

    print("[✔] Done! Check the file:", RESULT_FILE)

if __name__ == "__main__":
    main()
