# Deep Learning Toolbox

Easily create and run deep 
learning experiments using [Torch](http://torch.ch/) with minimal code.

Initially inspired by 
[ImageNet multi-GPU](https://github.com/soumith/imagenet-multiGPU.torch).

Similar frameworks:

* [dp](https://github.com/nicholas-leonard/dp)
* [torchnet](https://github.com/torchnet/torchnet) 

## Supports

* Multi-GPU implementation with automatic saving/loading of 
[models](doc/model.md).
* [Data](doc/data.md) iterator and [loaders](doc/loader.md) with 
multi-threading support.
* Multiple types of [training](doc/trainer.md) (simple, GAN, WGAN, BEGAN), 
with automatic checkpointing and logging of training loss.
* Easy [experiment creation](doc/trainer.md) and 
[dispatching](doc/dispatcher.md). Experiments are transferable accross 
machines (e.g. can start training on a GPU machine and finish on 
a non-GPU machine).
* [Slurm](doc/slurm.md) scheduler support for usage on HPC facilities.
* Settings parsing, optimizer, logging, colorspaces (XYZ, IPT, LMS, Lαβ). 
More info [here](doc/misc.md).
* Data loader interfaces for *MNIST*, *CIFAR*, *CelebA*, *Places*, *pix2pix*.
* Implementations of some (standard) models, including *LeNet5*, *VGG*, 
*AlexNet*, *Squeezenet*, *Colornet*, *UNET*.

## Installation

Make sure you have [Torch](http://torch.ch/) installed.

To install use:
```bash
git clone https://github.com/dmarnerides/dlt.git
cd dlt
./install.sh
```

## Warning / Disclaimer

I created this toolbox for my PhD, mostly to learn Lua, understand Torch in 
depth, and have a consistent workflow accross multiple machines and HPC 
facilities.

Only tested on Ubuntu and CentOS.

**If you use this package you will probably encounter bugs. 
If so please let me know!**

**Use at your own risk.**

If you use this code please cite the repo.

## Contact

dmarnerides@gmail.com