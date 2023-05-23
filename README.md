### Dependencies :
The main dependencies of the project are the following:
```yaml
python: 3.7.13
cuda: 11.3
torch: 1.12.1
```
You can set up a conda environment as follows
```
conda create --name=q2e python=3.7
conda activate q2e

conda update -n base -c defaults conda
conda install openblas-devel -c anaconda
conda install pytorch==1.12 torchvision torchaudio cudatoolkit=11.3 -c pytorch

pip install ninja==1.10.2.3
pip install json glob scipy yaml tqdm fire imageio wandb python-dotenv pyviz3d plyfile scikit-learn trimesh loguru albumentations volumentations
pip install antlr4-python3-runtime==4.8
pip install black==21.4b2
pip install omegaconf==2.0.6 hydra-core==1.0.5 --no-deps

pip install -U git+https://github.com/NVIDIA/MinkowskiEngine -v --no-deps --install-option="--blas_include_dirs=${CONDA_PREFIX}/include" --install-option="--blas=openblas"
cd third_party/pointnet2 && python setup.py install
```

### Data preprocessing :
After installing the dependencies, we preprocess the datasets.

#### ScanNet
The details of data processing of Scannet can refer to [Mask3D](https://github.com/JonasSchult/Mask3D).
```
python scannet/preprocess/scannet_preprocessing.py preprocess \
--data_dir="PATH_TO_RAW_SCANNET_DATASET" \
--save_dir="../../data/processed/scannet" \
--git_repo="PATH_TO_SCANNET_GIT_REPO" \
--scannet200=false
```

####  S3DIS
The details of data processing of S3DIS can refer to [One-Thing-One-Click](https://github.com/liuzhengzhe/One-Thing-One-Click).

```
cd s3dis/preprocess
python concate_new.py
python partition.py
python sup_voxel.py
```


### Inference :
Before inference, we must modify the path of **data_root** in val_scannet.py and val_s3dis.py to the path corresponding to validation set.
```
python val_scannet.py
python val_s3dis.py
```
### Model weights
[Model weights for ScanNet and S3DIS](https://mega.nz/folder/RC9FCayQ#tFGcAPzULFBJs6goycGDnw)
