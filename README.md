# RootDetective-DFC2023-Contest

## Installation 

**Step 1.** clone repository and update submodules

```
git clone https://github.com/efkandurakli/RoofDetective-DFC-2023.git 
cd RoofDetective-DFC-2023
git submodule update --init --remote --recursive
```


**Step 2.** create conda virtual environment and activate it 
```
conda create --name roof-detective-dfc2023 python=3.8
conda activate roof-detective-dfc2023
```

**Step 3.** install Pytorch following [official instructions](https://pytorch.org/get-started/locally/) 

**Step 4.** Install [MMCV](https://github.com/open-mmlab/mmcv) using [MIM](https://github.com/open-mmlab/mim) 
```
pip install -U openmim
mim install mmcv-full
```

**Step 5.** Install pycocotools, mmengine and MMDetection 
```
conda install -c conda-forge pycocotools
pip install mmengine
cd mmdetection
pip install -v -e .
```

**Step 6.** Install other required packages
```
pip install future tensorboard
conda install -c conda-forge tqdm
```

**Step 7.** Download config and checkpoint files and verify your installation 
```
mim download mmdet --config yolov3_mobilenetv2_320_300e_coco --dest .
python demo/image_demo.py demo/demo.jpg yolov3_mobilenetv2_320_300e_coco.py yolov3_mobilenetv2_320_300e_coco_20210719_215349-d18dff72.pth --device cpu --out-file result.jpg
```

You will see a new image `result.jpg` on your current folder, where bounding boxes are plotted on cars, benches, etc.

## Training

```
python tools/train.py $CONFIG --work-dir $CHECKPOINT_DIR
```

## Testing

```
python tools/test.py $CONFIG $checkpoint --format-only --eval-options "jsonfile_prefix=$SAVE_PATH"
```