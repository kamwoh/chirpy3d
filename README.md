# Chirpy3D: Creative Fine-grained 3D Object Fabrication via Part Sampling

### Installation

```
conda create -n chirpy3d python=3.10 --y
conda activate chirpy3d
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
pip install xformers==0.0.22.post7

pip install -r requirements.txt
```

### Dataset preparation

```
# cub200
cd src/chiryp3d/data/cub200_2011
wget https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz
tar -xvzf CUB_200_2011.tgz

# dog
cd src/chirpy3d/data/dogs
wget http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar
tar -xvf images.tar

# sims4-faces
cd src/chirpy3d/data/sims4-faces
wget https://huggingface.co/datasets/rocca/sims4-faces/resolve/main/fem.tar
wget https://huggingface.co/datasets/rocca/sims4-faces/resolve/main/masc.tar

mkdir fem && tar -xvf fem.tar -C fem
mkdir masc && tar -xvf masc.tar -C masc

# partimagenet
cd src/chirpy3d/data/PartImageNet
wget https://huggingface.co/datasets/kamwoh/partimagenet_preprocessed/resolve/main/partimagenet.zip
unzip partimagenet.zip
```

### Training

```
bash run_cub200.sh
bash run_dogs.sh
bash run_sims4.sh
bash run_partimagenet.sh
```

### Training Logs

Every epoch will run validation once to check the generated multiview images

```
tensorboard --logdir path/to/logs --port 6006 --host 0.0.0.0
```

### TODO

- [ ] Add dreamgaussian & threestudio for SDS
- [ ] Add pre-trained weights (I am retraining all models and upload huggingface)
- [ ] Add evaluation codes