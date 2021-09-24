""" Download datasets and prepare data """
import os
import urllib.request

download_root = "https://www.tnt.uni-hannover.de/de/project/MPT/data/MakingHigherOrderMOTScalable/"

data_dir = os.path.join("data")
result_dir = os.path.join("results")

datasets = ["MOT15", "MOT15-Preprocessed", "MOT17", "MOT17-Preprocessed", "MOT20", "MOT20-Preprocessed"]

''' Create directories '''
os.makedirs(result_dir, exist_ok=True)
os.makedirs(data_dir, exist_ok=True)
os.makedirs(os.path.join(data_dir, "tmp"), exist_ok=True)

''' Unzip pretrained models '''
model_container = os.path.join(data_dir, "models.zip")
if not os.path.exists(model_container):
    urllib.request.urlretrieve(download_root + "models.zip", model_container)
    os.system("unzip %s -d %s" % (model_container, data_dir))

''' Download datasets '''
for dataset in datasets:
    container = os.path.join(data_dir, "tmp", dataset + ".zip")
    download_link = download_root + dataset + ".zip"
    if not os.path.exists(container):
        os.system("wget -O %s %s" % (container, download_link))
        #urllib.request.urlretrieve(download_link, container)

''' Unzip datasets '''
for dataset in datasets:
    container = os.path.join(data_dir, "tmp", dataset + ".zip")
    dst = os.path.join(data_dir, "tmp")
    os.system("unzip %s -d %s" % (container, dst))

