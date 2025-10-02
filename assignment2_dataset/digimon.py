import requests
from bs4 import BeautifulSoup
import os

url = "https://wikimon.net/Visual_List_of_Digimon"
save_dir = "./not_pokemon"
os.makedirs(save_dir, exist_ok=True)

# Get page
response = requests.get(url)
soup = BeautifulSoup(response.text, "html.parser")

# Find all Digimon images
images = soup.find_all("img")

count = 0
for img in images:
    img_url = img.get("src")
    if not img_url:
        continue
    if img_url.startswith("//"):
        img_url = "https:" + img_url  # fix protocol
    elif img_url.startswith("/"):
        img_url = "https://wikimon.net" + img_url

    try:
        img_data = requests.get(img_url).content
        filename = os.path.join(save_dir, f"digimon_{count}.png")
        with open(filename, "wb") as f:
            f.write(img_data)
        count += 1
    except:
        continue

print(f"Downloaded {count} Digimon images to {save_dir}")
