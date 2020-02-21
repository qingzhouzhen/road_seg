# pytorch FCN road segmentation

这是一个demo工程，试图从无人机上分割出道路位置

## 运行环境

* ubuntu16.4
* CUDA 9.0
* Anaconda 3 （numpy、os、datetime、matplotlib）
* pytorch == 0.4.1 or 1.0
* torchvision == 0.2.1
* OpenCV-Python == 3.4.1

## 训练

```sh
python train.py
```

训练从原图根据labeme标注的文件生成黑白的二值标签文件，原图和标签图片太大，没有push, 转换脚本参考：

```
https://github.com/qingzhouzhen/data_process/blob/master/generate_road_mask.py
```

## 测试

```
python demo.py
```

如果没有出错，应该出现如下结果

![](./assets/DJI_0510_10.jpg)

参考：<https://github.com/bat67/pytorch-FCN-easiest-demo>

