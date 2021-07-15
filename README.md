# High-Capacity Expert Binary Networks (ICLR 2021)

This code provides the core components for building networks based on the architectures and Expert Binary Convolutional Block introduced in the _High-Capacity Expert Binary Networks_ paper. You can find the full version of the paper [here](https://arxiv.org/pdf/2010.03558).

## 1. Installation

To install and test the code simply clone the current repo, install the required packages listed bellow and prepare the training and/or testing(validation) data.

### 1.1 Requirements
torch >= 1.6.0
torchvision >= 0.5.0
bnn >= 0.1.1

### 1.2. Data preparation
 Download the ImageNet dataset from the official [webpage](http://image-net.org/download-images), creating a folder with the following structure: 
```
│imagenet/
├──train/
│  ├── n01440764
│  │   ├── n01440764_10026.JPEG
│  │   ├── n01440764_10027.JPEG
│  │   ├── ......
│  ├── ......
├──val/
│  ├── n01440764
│  │   ├── ILSVRC2012_val_00000293.JPEG
│  │   ├── ILSVRC2012_val_00002138.JPEG
│  │   ├── ......
│  ├── ......
```
You can construct this structure using for example the script found [here](https://gist.github.com/BIGBALLON/8a71d225eff18d88e469e6ea9b39cef4).

## 2. Testing pretrained model

A packed model, with the extra training components removed (i.e. the gate selection is performed using selection instead of multiplication as its the case for training etc) is available for download [here](https://www.adrianbulat.com/downloads/ICLR2021/model_binary_ebresnet.pth.tar).

```
python test/test.py imagenet_valid_location --path-to-model download_model_path
```

## Reference

If you find this repo useful, please consider citing:
```
@inproceedings{bulat2021high,
  title={High-Capacity Expert Binary Networks},
  author={Bulat, Adrian and Martinez, Brais and Tzimiropoulos, Georgios},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2021}
}
```
