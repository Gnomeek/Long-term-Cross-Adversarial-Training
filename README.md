# Long-term Cross Adversarial Training
## Structure
1. Few-shot-datasets folder: store datasets including `MiniImageNet`, `TieredImageNet`, `CIFAR-FS`
2. data folder: codes to preprocess and load dataset
3. models folder and qpth model: embedding network
4. experiments folde: store checkpoints, test results, log file, figures
4. test.py: test model
5. train.py: train model
6. utils.py: utilities

## Requirements
- Pytorch
- Python
- CUDA
- Numpy
- Matplotlib

## Credits
- Goldblum, M., Fowl, L., and Goldstein, T. Adversariallyrobust few-shot learning: A meta-learning approach.Advances in Neural Information Processing Systems, 33,2020.
- Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun Deep Residual Learning for Image Recognition. arXiv:1512.03385
- Cihang Xie, Yuxin Wu, Laurens van der Maaten, Alan Yuille, Kaiming He Feature Denoising for Improving Adversarial Robustness. arXiv:1812.03411

## License
- MIT License