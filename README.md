# CDV-SLAM: Compact Deep Visual SLAM via Unified Semantic and Geometric perception
## Setup
The code was tested on Ubuntu>=18.04 and Windows 10.</br>

Clone the repo
```
git clone https://github.com/FrankYard/CDV-SLAM
cd CDV-SLAM
```
Create and activate the anaconda environment
```
conda env create -f environment.yml
conda activate cdv-slam
```

Next install the cdvslam package with cuda extension
```bash
wget https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip
unzip eigen-3.4.0.zip -d thirdparty

# install CDVO
pip install .
```
To download the model parameters, use the following link [Dropbox](https://www.dropbox.com/scl/fo/frlx9r8wy3zb01eqtb5k0/ADoKPJcohVUpz_6gEMaUec0?rlkey=h54d7dxmocysum8efcpeckpa5&st=ipzsphh1&dl=0).

(Optional)The code is backword compatible with DPVO/DPV-SLAM. If you what to use DPVO/DPV-SLAM, run the following:
```bash
# download models and data (~2GB)
./download_dpvo_models_and_data.sh
```

### Classical Backend (optional)

The classical backend for closing very large loops is adopted from DPV-SLAM, which requires extra installation.

Step 1. Install the OpenCV C++ API. On Ubuntu, you can use
```bash
sudo apt-get install -y libopencv-dev
```
Step 2. Install DBoW2
```bash
git clone https://github.com/lahavlipson/DBoW2
cd DBoW2
mkdir -p build && cd build
cmake .. # tested with cmake 3.22.1 and gcc/cc 11.4.0 on Ubuntu
make # tested with GNU Make 4.3
sudo make install
cd ../..
```

Step 3. Install the image retrieval
```bash
pip install ./DPRetrieval
```

## Demos
CDV-SLAM
Use '--version' to switch between DPVO/DPV-SLAM and CDVO/CDV-SLAM. By default use CDVO/CDV-SLAM.

```bash
python demo.py \
    --imagedir=<path to image directory or video> \
    --calib=<path to calibration file> \
    --viz # enable visualization
    --plot # save trajectory plot
    --save_ply # save point cloud as a .ply file
    --save_trajectory # save the predicted trajectory as .txt in TUM format
    --save_colmap # save point cloud + trajectory in the standard COLMAP text format
```

### iPhone
```bash
python demo.py --imagedir=movies/IMG_0492.MOV --calib=calib/iphone.txt --stride=5 --plot --viz
```

### TartanAir
Download a sequence from [TartanAir](https://theairlab.org/tartanair-dataset/) (several samples are availabe from download directly from the webpage)
```bash
python demo.py --imagedir=<path to image_left> --calib=calib/tartan.txt --stride=1 --plot --viz
```

### EuRoC
Download a sequence from [EuRoC](https://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets) (download ASL format)
```bash
python demo.py --imagedir=<path to mav0/cam0/data/> --calib=calib/euroc.txt --stride=2 --plot --viz
```

## SLAM Backends
To run DPVO with a SLAM backend (i.e., DPV-SLAM), add
```bash
--opts LOOP_CLOSURE True
```
to any `evaluate_X.py` script or to `demo.py`

If installed, the classical backend can also be enabled using 
```
--opts CLASSIC_LOOP_CLOSURE True
```

## Evaluation
We provide evaluation scripts for TartanAir, EuRoC, TUM-RGBD and ICL-NUIM. Up to date result logs on these datasets can be found in the `logs` directory.

### TartanAir:
Results on the validation split and test set can be obtained with the command:
```
python evaluate_tartan.py --trials=5 --split=validation --plot --save_trajectory
```

### EuRoC:
```
python evaluate_euroc.py --trials=5 --plot --save_trajectory
```

### TUM-RGBD:
```
python evaluate_tum.py --trials=5 --plot --save_trajectory
```

### ICL-NUIM:
```
python evaluate_icl_nuim.py --trials=5 --plot --save_trajectory
```

### KITTI:
```
python evaluate_kitti.py --trials=5 --plot --save_trajectory
```

## Training
Make sure you have run `./download_models_and_data.sh`. Your directory structure should look as follows

```Shell
├── datasets
    ├── TartanAir.pickle
    ├── TartanAir
        ├── abandonedfactory
        ├── abandonedfactory_night
        ├── ...
        ├── westerndesert
    ...
```

To train (log files will be written to `runs/<your name>`). Model will be run on the validation split every 10k iterations
```
python train_cdvo.py --steps=240000 --lr=0.00008 --name=<your name>
```

## Acknowledgements
This project is built upon
[DPVO](https://github.com/princeton-vl/DPVO).
We use parts of code from
[Dinov2](https://github.com/facebookresearch/dinov2),
[XFeat](https://github.com/verlab/accelerated_features),
[Mickey](https://github.com/nianticlabs/mickey),
and [LightGlue](https://github.com/cvg/LightGlue).
