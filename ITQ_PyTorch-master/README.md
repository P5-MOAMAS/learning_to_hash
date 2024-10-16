# A pytorch implementation of ITQ

### DISCLAIMER
This code is based on the code from the paper:
"Iterative Quantization: A Procrustean Approach to Learning Binary Codes
for Large-scale Image Retrieval" TPAMI-2013. The code has been modified to fit the criteria of this project.
## REQUIREMENTS
`pip install -r requirements.txt`

1. pytorch >= 1.0
2. loguru

## Features
1. Mnist
2. Cifar10
3. Imagenet100

## USAGE
```
usage: run.py [-h] [--feature-set FEATURE] [--root ROOT]
              [--binary-hashcode-length CODE_LENGTH] [--max-iter MAX_ITER] [--topk TOPK]
              [--gpu GPU]

ITQ_PyTorch

optional arguments:
  -h, --help            show this help message and exit
  --dataset DATASET     Dataset name.
  --root ROOT           Path of dataset
  --code-length CODE_LENGTH
                        Binary hash code length.(default:
                        8,16,24,32,48,64,96,128)
  --max-iter MAX_ITER   Number of iterations.(default: 3)
  --topk TOPK           Calculate map of top k.(default: ALL)
  --gpu GPU             Using gpu.(default: False)
```

## EXPERIMENTS
To be filled out