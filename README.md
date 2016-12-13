EECS 545 Course Project: Image Generation and Unsupervised Learning with Deep Convolutional Generative Adversarial Networks
===========================================================================================================================


## Commandline Options

`
python main.py -h
`

## Training

```
python main.py --mode train --dataset mnist 
python main.py --mode train --dataset stl10
```

By the end of each epoch, a model checkpoint is saved to `./checkpoints`, and samples are saved to `./samples`


## Sampling

`
python main.py --mode sample --dataset mnist --checkpoint /path/to/model --output results.png
`

