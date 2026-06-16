```
git clone --branch main --single-branch git@github.com:HuMathe/sonoworld.git
cd sonoworld
```

```
conda create -n sonoworld python=3.12
conda activate sonoworld
pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/cu128

mkdir third_party
cd third_party
git clone https://github.com/facebookresearch/sam3.git
cd sam3
git checkout 757bbb0206a0b68bee81b17d7eb4877177025b2f
pip install -e .

git clone https://github.com/hkchengrex/MMAudio.git
cd MMAudio
pip install -e .
cd ..

git clone https://github.com/cvg/GeoCalib.git
cd GeoCalib
pip install -e .
cd ..

pip install git+https://github.com/microsoft/MoGe.git

cd ..
pip install -r requirements.txt
pip install --force-reinstall "setuptools<82"
```

