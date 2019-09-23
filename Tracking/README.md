This part of code was adapted from [offical release of SiamMask](https://github.com/foolwood/SiamMask). Please refer to this repo before use.

You may follow the **Environment setup** or run
```
conda env create -f environment.yml
conda activate sky
bash make.sh
```
to setup the environment.

Follow the original code to setup datasets [Youtube-VOS](https://youtube-vos.org/dataset/download/), 
[COCO](http://cocodataset.org/#download), 
[ImageNet-DET](http://image-net.org/challenges/LSVRC/2015/), 
and [ImageNet-VID](http://image-net.org/challenges/LSVRC/2015/) and train/test on them. 
An example to train SkyNet on Youtube-VOS:
```
python tools/train_siammask.py --config=config_skynet.json -b 32 -j 16 --epochs 60 --log logs/log.txt --save_dir=snapshot_sky
```
To test on [GOT-10k](http://got-10k.aitestunion.com/) dataset, refer [test_got10k.py](./test_got10k.py).

## License
Licensed under an MIT license.

