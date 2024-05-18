REQUIREMENTS
PyTorch and timm 0.6.11 (pip install timm==0.6.11).

Data preparation: ImageNet with the following folder structure, you can extract ImageNet by this script.

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

For Training..
We use batch size of 4096 by default and we show how to train models with 8 GPUs. For multi-node training, adjust --grad-accum-steps according to your situations.

DATA_PATH=/path/to/imagenet
CODE_PATH=/path/to/code/GHCS # modify code path here


ALL_BATCH_SIZE=4096
NUM_GPU=8
GRAD_ACCUM_STEPS=4 # Adjust according to your GPU numbers and memory size.
let BATCH_SIZE=ALL_BATCH_SIZE/NUM_GPU/GRAD_ACCUM_STEPS


MODEL=mambaout_tiny 
DROP_PATH=0.2


cd $CODE_PATH && sh distributed_train.sh $NUM_GPU $DATA_PATH \
--model $MODEL --opt adamw --lr 4e-3 --warmup-epochs 20 \
-b $BATCH_SIZE --grad-accum-steps $GRAD_ACCUM_STEPS \
--drop-path $DROP_PATH



To evaluate models, run:

MODEL=validate
python3 validate.py /path/to/imagenet  --model $MODEL -b 128 \
  --pretrained

To evaluate models, run:

Scripts=test
python3 test.py /path/to/imagenet  --model $MODEL -b 128 \
  --pretrained