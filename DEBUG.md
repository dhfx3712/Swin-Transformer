(tf2.7) admin@bogon Swin-Transformer % python main.py --cfg configs/swin_base_patch4_window7_224.yaml --local_rank -1 --data-path /Users/admin/data/cifar100/data
=> merge config from configs/swin_base_patch4_window7_224.yaml
[2022-11-03 10:20:14 swin_base_patch4_window7_224](main.py 369): INFO AMP_OPT_LEVEL: O0
AUG:
  AUTO_AUGMENT: rand-m9-mstd0.5-inc1
  COLOR_JITTER: 0.4
  CUTMIX: 1.0
  CUTMIX_MINMAX: null
  MIXUP: 0.8
  MIXUP_MODE: batch
  MIXUP_PROB: 1.0
  MIXUP_SWITCH_PROB: 0.5
  RECOUNT: 1
  REMODE: pixel
  REPROB: 0.25
BASE:
- ''
DATA:
  BATCH_SIZE: 128
  CACHE_MODE: part
  DATASET: imagenet
  DATA_PATH: /Users/admin/data/cifar100/data
  IMG_SIZE: 224
  INTERPOLATION: bicubic
  NUM_WORKERS: 8
  PIN_MEMORY: true
  ZIP_MODE: false
EVAL_MODE: false
LOCAL_RANK: -1
MODEL:
  DROP_PATH_RATE: 0.5
  DROP_RATE: 0.0
  LABEL_SMOOTHING: 0.1
  NAME: swin_base_patch4_window7_224
  NUM_CLASSES: 100
  PRETRAINED: ''
  RESUME: ''
  SWIN:
    APE: false
    DEPTHS:
    - 2
    - 2
    - 18
    - 2
    EMBED_DIM: 128
    IN_CHANS: 3
    MLP_RATIO: 4.0
    NUM_HEADS:
    - 4
    - 8
    - 16
    - 32
    PATCH_NORM: true
    PATCH_SIZE: 4
    QKV_BIAS: true
    QK_SCALE: null
    WINDOW_SIZE: 7
  SWIN_MLP:
    APE: false
    DEPTHS:
    - 2
    - 2
    - 6
    - 2
    EMBED_DIM: 96
    IN_CHANS: 3
    MLP_RATIO: 4.0
    NUM_HEADS:
    - 3
    - 6
    - 12
    - 24
    PATCH_NORM: true
    PATCH_SIZE: 4
    WINDOW_SIZE: 7
  TYPE: swin
OUTPUT: output/swin_base_patch4_window7_224/default
PRINT_FREQ: 10
SAVE_FREQ: 1
SEED: 0
TAG: default
TEST:
  CROP: true
  SEQUENTIAL: false
THROUGHPUT_MODE: false
TRAIN:
  ACCUMULATION_STEPS: 0
  AUTO_RESUME: true
  BASE_LR: 0.000125
  CLIP_GRAD: 5.0
  EPOCHS: 300
  LR_SCHEDULER:
    DECAY_EPOCHS: 30
    DECAY_RATE: 0.1
    NAME: cosine
  MIN_LR: 1.25e-06
  OPTIMIZER:
    BETAS:
    - 0.9
    - 0.999
    EPS: 1.0e-08
    MOMENTUM: 0.9
    NAME: adamw
  START_EPOCH: 0
  USE_CHECKPOINT: false
  WARMUP_EPOCHS: 20
  WARMUP_LR: 1.25e-07
  WEIGHT_DECAY: 0.05

aaa
root : /Users/admin/data/cifar100/data Dataset ImageFolder
    Number of datapoints: 60000
    Root location: /Users/admin/data/cifar100/data
    StandardTransform
Transform: Compose(
               RandomResizedCropAndInterpolation(size=(224, 224), scale=(0.08, 1.0), ratio=(0.75, 1.3333), interpolation=bicubic)
               RandomHorizontalFlip(p=0.5)
               RandAugment(n=2, ops=
                AugmentOp(name=AutoContrast, p=0.5, m=9, mstd=0.5)
                AugmentOp(name=Equalize, p=0.5, m=9, mstd=0.5)
                AugmentOp(name=Invert, p=0.5, m=9, mstd=0.5)
                AugmentOp(name=Rotate, p=0.5, m=9, mstd=0.5)
                AugmentOp(name=PosterizeIncreasing, p=0.5, m=9, mstd=0.5)
                AugmentOp(name=SolarizeIncreasing, p=0.5, m=9, mstd=0.5)
                AugmentOp(name=SolarizeAdd, p=0.5, m=9, mstd=0.5)
                AugmentOp(name=ColorIncreasing, p=0.5, m=9, mstd=0.5)
                AugmentOp(name=ContrastIncreasing, p=0.5, m=9, mstd=0.5)
                AugmentOp(name=BrightnessIncreasing, p=0.5, m=9, mstd=0.5)
                AugmentOp(name=SharpnessIncreasing, p=0.5, m=9, mstd=0.5)
                AugmentOp(name=ShearX, p=0.5, m=9, mstd=0.5)
                AugmentOp(name=ShearY, p=0.5, m=9, mstd=0.5)
                AugmentOp(name=TranslateXRel, p=0.5, m=9, mstd=0.5)
                AugmentOp(name=TranslateYRel, p=0.5, m=9, mstd=0.5))
               ToTensor()
               Normalize(mean=tensor([0.4850, 0.4560, 0.4060]), std=tensor([0.2290, 0.2240, 0.2250]))
               RandomErasing(p=0.25, mode=pixel, count=(1, 1))
           )
aaa
root : /Users/admin/data/cifar100/data Dataset ImageFolder
    Number of datapoints: 60000
    Root location: /Users/admin/data/cifar100/data
    StandardTransform
Transform: Compose(
               Resize(size=256, interpolation=bicubic, max_size=None, antialias=None)
               CenterCrop(size=(224, 224))
               ToTensor()
               Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
           )
[2022-11-03 10:20:14 swin_base_patch4_window7_224](main.py 84): INFO Creating model:swin/swin_base_patch4_window7_224
/Users/admin/opt/anaconda3/envs/tf2.7/lib/python3.7/site-packages/torch/functional.py:568: UserWarning: torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument. (Triggered internally at  /Users/distiller/project/pytorch/aten/src/ATen/native/TensorShape.cpp:2228.)
  return _VF.meshgrid(tensors, **kwargs)  # type: ignore[attr-defined]
[2022-11-03 10:20:18 swin_base_patch4_window7_224](main.py 87): INFO SwinTransformer(
  (patch_embed): PatchEmbed(
    (proj): Conv2d(3, 128, kernel_size=(4, 4), stride=(4, 4))
    (norm): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
  )
  (pos_drop): Dropout(p=0.0, inplace=False)
  (layers): ModuleList(
    (0): BasicLayer(
      dim=128, input_resolution=(56, 56), depth=2
      (blocks): ModuleList(
        (0): SwinTransformerBlock(
          dim=128, input_resolution=(56, 56), num_heads=4, window_size=7, shift_size=0, mlp_ratio=4.0
          (norm1): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
          (attn): WindowAttention(
            dim=128, window_size=(7, 7), num_heads=4
            (qkv): Linear(in_features=128, out_features=384, bias=True)
            (attn_drop): Dropout(p=0.0, inplace=False)
            (proj): Linear(in_features=128, out_features=128, bias=True)
            (proj_drop): Dropout(p=0.0, inplace=False)
            (softmax): Softmax(dim=-1)
          )
          (drop_path): Identity()
          (norm2): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
          (mlp): Mlp(
            (fc1): Linear(in_features=128, out_features=512, bias=True)
            (act): GELU()
            (fc2): Linear(in_features=512, out_features=128, bias=True)
            (drop): Dropout(p=0.0, inplace=False)
          )
        )
        (1): SwinTransformerBlock(
          dim=128, input_resolution=(56, 56), num_heads=4, window_size=7, shift_size=3, mlp_ratio=4.0
          (norm1): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
          (attn): WindowAttention(
            dim=128, window_size=(7, 7), num_heads=4
            (qkv): Linear(in_features=128, out_features=384, bias=True)
            (attn_drop): Dropout(p=0.0, inplace=False)
            (proj): Linear(in_features=128, out_features=128, bias=True)
            (proj_drop): Dropout(p=0.0, inplace=False)
            (softmax): Softmax(dim=-1)
          )
          (drop_path): DropPath()
          (norm2): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
          (mlp): Mlp(
            (fc1): Linear(in_features=128, out_features=512, bias=True)
            (act): GELU()
            (fc2): Linear(in_features=512, out_features=128, bias=True)
            (drop): Dropout(p=0.0, inplace=False)
          )
        )
      )
      (downsample): PatchMerging(
        input_resolution=(56, 56), dim=128
        (reduction): Linear(in_features=512, out_features=256, bias=False)
        (norm): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
      )
    )
    (1): BasicLayer(
      dim=256, input_resolution=(28, 28), depth=2
      (blocks): ModuleList(
        (0): SwinTransformerBlock(
          dim=256, input_resolution=(28, 28), num_heads=8, window_size=7, shift_size=0, mlp_ratio=4.0
          (norm1): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
          (attn): WindowAttention(
            dim=256, window_size=(7, 7), num_heads=8
            (qkv): Linear(in_features=256, out_features=768, bias=True)
            (attn_drop): Dropout(p=0.0, inplace=False)
            (proj): Linear(in_features=256, out_features=256, bias=True)
            (proj_drop): Dropout(p=0.0, inplace=False)
            (softmax): Softmax(dim=-1)
          )
          (drop_path): DropPath()
          (norm2): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
          (mlp): Mlp(
            (fc1): Linear(in_features=256, out_features=1024, bias=True)
            (act): GELU()
            (fc2): Linear(in_features=1024, out_features=256, bias=True)
            (drop): Dropout(p=0.0, inplace=False)
          )
        )
        (1): SwinTransformerBlock(
          dim=256, input_resolution=(28, 28), num_heads=8, window_size=7, shift_size=3, mlp_ratio=4.0
          (norm1): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
          (attn): WindowAttention(
            dim=256, window_size=(7, 7), num_heads=8
            (qkv): Linear(in_features=256, out_features=768, bias=True)
            (attn_drop): Dropout(p=0.0, inplace=False)
            (proj): Linear(in_features=256, out_features=256, bias=True)
            (proj_drop): Dropout(p=0.0, inplace=False)
            (softmax): Softmax(dim=-1)
          )
          (drop_path): DropPath()
          (norm2): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
          (mlp): Mlp(
            (fc1): Linear(in_features=256, out_features=1024, bias=True)
            (act): GELU()
            (fc2): Linear(in_features=1024, out_features=256, bias=True)
            (drop): Dropout(p=0.0, inplace=False)
          )
        )
      )
      (downsample): PatchMerging(
        input_resolution=(28, 28), dim=256
        (reduction): Linear(in_features=1024, out_features=512, bias=False)
        (norm): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
      )
    )
    (2): BasicLayer(
      dim=512, input_resolution=(14, 14), depth=18
      (blocks): ModuleList(
        (0): SwinTransformerBlock(
          dim=512, input_resolution=(14, 14), num_heads=16, window_size=7, shift_size=0, mlp_ratio=4.0
          (norm1): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
          (attn): WindowAttention(
            dim=512, window_size=(7, 7), num_heads=16
            (qkv): Linear(in_features=512, out_features=1536, bias=True)
            (attn_drop): Dropout(p=0.0, inplace=False)
            (proj): Linear(in_features=512, out_features=512, bias=True)
            (proj_drop): Dropout(p=0.0, inplace=False)
            (softmax): Softmax(dim=-1)
          )
          (drop_path): DropPath()
          (norm2): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
          (mlp): Mlp(
            (fc1): Linear(in_features=512, out_features=2048, bias=True)
            (act): GELU()
            (fc2): Linear(in_features=2048, out_features=512, bias=True)
            (drop): Dropout(p=0.0, inplace=False)
          )
        )
        (1): SwinTransformerBlock(
          dim=512, input_resolution=(14, 14), num_heads=16, window_size=7, shift_size=3, mlp_ratio=4.0
          (norm1): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
          (attn): WindowAttention(
            dim=512, window_size=(7, 7), num_heads=16
            (qkv): Linear(in_features=512, out_features=1536, bias=True)
            (attn_drop): Dropout(p=0.0, inplace=False)
            (proj): Linear(in_features=512, out_features=512, bias=True)
            (proj_drop): Dropout(p=0.0, inplace=False)
            (softmax): Softmax(dim=-1)
          )
          (drop_path): DropPath()
          (norm2): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
          (mlp): Mlp(
            (fc1): Linear(in_features=512, out_features=2048, bias=True)
            (act): GELU()
            (fc2): Linear(in_features=2048, out_features=512, bias=True)
            (drop): Dropout(p=0.0, inplace=False)
          )
        )
        (2): SwinTransformerBlock(
          dim=512, input_resolution=(14, 14), num_heads=16, window_size=7, shift_size=0, mlp_ratio=4.0
          (norm1): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
          (attn): WindowAttention(
            dim=512, window_size=(7, 7), num_heads=16
            (qkv): Linear(in_features=512, out_features=1536, bias=True)
            (attn_drop): Dropout(p=0.0, inplace=False)
            (proj): Linear(in_features=512, out_features=512, bias=True)
            (proj_drop): Dropout(p=0.0, inplace=False)
            (softmax): Softmax(dim=-1)
          )
          (drop_path): DropPath()
          (norm2): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
          (mlp): Mlp(
            (fc1): Linear(in_features=512, out_features=2048, bias=True)
            (act): GELU()
            (fc2): Linear(in_features=2048, out_features=512, bias=True)
            (drop): Dropout(p=0.0, inplace=False)
          )
        )
        (3): SwinTransformerBlock(
          dim=512, input_resolution=(14, 14), num_heads=16, window_size=7, shift_size=3, mlp_ratio=4.0
          (norm1): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
          (attn): WindowAttention(
            dim=512, window_size=(7, 7), num_heads=16
            (qkv): Linear(in_features=512, out_features=1536, bias=True)
            (attn_drop): Dropout(p=0.0, inplace=False)
            (proj): Linear(in_features=512, out_features=512, bias=True)
            (proj_drop): Dropout(p=0.0, inplace=False)
            (softmax): Softmax(dim=-1)
          )
          (drop_path): DropPath()
          (norm2): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
          (mlp): Mlp(
            (fc1): Linear(in_features=512, out_features=2048, bias=True)
            (act): GELU()
            (fc2): Linear(in_features=2048, out_features=512, bias=True)
            (drop): Dropout(p=0.0, inplace=False)
          )
        )
        (4): SwinTransformerBlock(
          dim=512, input_resolution=(14, 14), num_heads=16, window_size=7, shift_size=0, mlp_ratio=4.0
          (norm1): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
          (attn): WindowAttention(
            dim=512, window_size=(7, 7), num_heads=16
            (qkv): Linear(in_features=512, out_features=1536, bias=True)
            (attn_drop): Dropout(p=0.0, inplace=False)
            (proj): Linear(in_features=512, out_features=512, bias=True)
            (proj_drop): Dropout(p=0.0, inplace=False)
            (softmax): Softmax(dim=-1)
          )
          (drop_path): DropPath()
          (norm2): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
          (mlp): Mlp(
            (fc1): Linear(in_features=512, out_features=2048, bias=True)
            (act): GELU()
            (fc2): Linear(in_features=2048, out_features=512, bias=True)
            (drop): Dropout(p=0.0, inplace=False)
          )
        )
        (5): SwinTransformerBlock(
          dim=512, input_resolution=(14, 14), num_heads=16, window_size=7, shift_size=3, mlp_ratio=4.0
          (norm1): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
          (attn): WindowAttention(
            dim=512, window_size=(7, 7), num_heads=16
            (qkv): Linear(in_features=512, out_features=1536, bias=True)
            (attn_drop): Dropout(p=0.0, inplace=False)
            (proj): Linear(in_features=512, out_features=512, bias=True)
            (proj_drop): Dropout(p=0.0, inplace=False)
            (softmax): Softmax(dim=-1)
          )
          (drop_path): DropPath()
          (norm2): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
          (mlp): Mlp(
            (fc1): Linear(in_features=512, out_features=2048, bias=True)
            (act): GELU()
            (fc2): Linear(in_features=2048, out_features=512, bias=True)
            (drop): Dropout(p=0.0, inplace=False)
          )
        )
        (6): SwinTransformerBlock(
          dim=512, input_resolution=(14, 14), num_heads=16, window_size=7, shift_size=0, mlp_ratio=4.0
          (norm1): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
          (attn): WindowAttention(
            dim=512, window_size=(7, 7), num_heads=16
            (qkv): Linear(in_features=512, out_features=1536, bias=True)
            (attn_drop): Dropout(p=0.0, inplace=False)
            (proj): Linear(in_features=512, out_features=512, bias=True)
            (proj_drop): Dropout(p=0.0, inplace=False)
            (softmax): Softmax(dim=-1)
          )
          (drop_path): DropPath()
          (norm2): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
          (mlp): Mlp(
            (fc1): Linear(in_features=512, out_features=2048, bias=True)
            (act): GELU()
            (fc2): Linear(in_features=2048, out_features=512, bias=True)
            (drop): Dropout(p=0.0, inplace=False)
          )
        )
        (7): SwinTransformerBlock(
          dim=512, input_resolution=(14, 14), num_heads=16, window_size=7, shift_size=3, mlp_ratio=4.0
          (norm1): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
          (attn): WindowAttention(
            dim=512, window_size=(7, 7), num_heads=16
            (qkv): Linear(in_features=512, out_features=1536, bias=True)
            (attn_drop): Dropout(p=0.0, inplace=False)
            (proj): Linear(in_features=512, out_features=512, bias=True)
            (proj_drop): Dropout(p=0.0, inplace=False)
            (softmax): Softmax(dim=-1)
          )
          (drop_path): DropPath()
          (norm2): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
          (mlp): Mlp(
            (fc1): Linear(in_features=512, out_features=2048, bias=True)
            (act): GELU()
            (fc2): Linear(in_features=2048, out_features=512, bias=True)
            (drop): Dropout(p=0.0, inplace=False)
          )
        )
        (8): SwinTransformerBlock(
          dim=512, input_resolution=(14, 14), num_heads=16, window_size=7, shift_size=0, mlp_ratio=4.0
          (norm1): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
          (attn): WindowAttention(
            dim=512, window_size=(7, 7), num_heads=16
            (qkv): Linear(in_features=512, out_features=1536, bias=True)
            (attn_drop): Dropout(p=0.0, inplace=False)
            (proj): Linear(in_features=512, out_features=512, bias=True)
            (proj_drop): Dropout(p=0.0, inplace=False)
            (softmax): Softmax(dim=-1)
          )
          (drop_path): DropPath()
          (norm2): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
          (mlp): Mlp(
            (fc1): Linear(in_features=512, out_features=2048, bias=True)
            (act): GELU()
            (fc2): Linear(in_features=2048, out_features=512, bias=True)
            (drop): Dropout(p=0.0, inplace=False)
          )
        )
        (9): SwinTransformerBlock(
          dim=512, input_resolution=(14, 14), num_heads=16, window_size=7, shift_size=3, mlp_ratio=4.0
          (norm1): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
          (attn): WindowAttention(
            dim=512, window_size=(7, 7), num_heads=16
            (qkv): Linear(in_features=512, out_features=1536, bias=True)
            (attn_drop): Dropout(p=0.0, inplace=False)
            (proj): Linear(in_features=512, out_features=512, bias=True)
            (proj_drop): Dropout(p=0.0, inplace=False)
            (softmax): Softmax(dim=-1)
          )
          (drop_path): DropPath()
          (norm2): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
          (mlp): Mlp(
            (fc1): Linear(in_features=512, out_features=2048, bias=True)
            (act): GELU()
            (fc2): Linear(in_features=2048, out_features=512, bias=True)
            (drop): Dropout(p=0.0, inplace=False)
          )
        )
        (10): SwinTransformerBlock(
          dim=512, input_resolution=(14, 14), num_heads=16, window_size=7, shift_size=0, mlp_ratio=4.0
          (norm1): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
          (attn): WindowAttention(
            dim=512, window_size=(7, 7), num_heads=16
            (qkv): Linear(in_features=512, out_features=1536, bias=True)
            (attn_drop): Dropout(p=0.0, inplace=False)
            (proj): Linear(in_features=512, out_features=512, bias=True)
            (proj_drop): Dropout(p=0.0, inplace=False)
            (softmax): Softmax(dim=-1)
          )
          (drop_path): DropPath()
          (norm2): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
          (mlp): Mlp(
            (fc1): Linear(in_features=512, out_features=2048, bias=True)
            (act): GELU()
            (fc2): Linear(in_features=2048, out_features=512, bias=True)
            (drop): Dropout(p=0.0, inplace=False)
          )
        )
        (11): SwinTransformerBlock(
          dim=512, input_resolution=(14, 14), num_heads=16, window_size=7, shift_size=3, mlp_ratio=4.0
          (norm1): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
          (attn): WindowAttention(
            dim=512, window_size=(7, 7), num_heads=16
            (qkv): Linear(in_features=512, out_features=1536, bias=True)
            (attn_drop): Dropout(p=0.0, inplace=False)
            (proj): Linear(in_features=512, out_features=512, bias=True)
            (proj_drop): Dropout(p=0.0, inplace=False)
            (softmax): Softmax(dim=-1)
          )
          (drop_path): DropPath()
          (norm2): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
          (mlp): Mlp(
            (fc1): Linear(in_features=512, out_features=2048, bias=True)
            (act): GELU()
            (fc2): Linear(in_features=2048, out_features=512, bias=True)
            (drop): Dropout(p=0.0, inplace=False)
          )
        )
        (12): SwinTransformerBlock(
          dim=512, input_resolution=(14, 14), num_heads=16, window_size=7, shift_size=0, mlp_ratio=4.0
          (norm1): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
          (attn): WindowAttention(
            dim=512, window_size=(7, 7), num_heads=16
            (qkv): Linear(in_features=512, out_features=1536, bias=True)
            (attn_drop): Dropout(p=0.0, inplace=False)
            (proj): Linear(in_features=512, out_features=512, bias=True)
            (proj_drop): Dropout(p=0.0, inplace=False)
            (softmax): Softmax(dim=-1)
          )
          (drop_path): DropPath()
          (norm2): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
          (mlp): Mlp(
            (fc1): Linear(in_features=512, out_features=2048, bias=True)
            (act): GELU()
            (fc2): Linear(in_features=2048, out_features=512, bias=True)
            (drop): Dropout(p=0.0, inplace=False)
          )
        )
        (13): SwinTransformerBlock(
          dim=512, input_resolution=(14, 14), num_heads=16, window_size=7, shift_size=3, mlp_ratio=4.0
          (norm1): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
          (attn): WindowAttention(
            dim=512, window_size=(7, 7), num_heads=16
            (qkv): Linear(in_features=512, out_features=1536, bias=True)
            (attn_drop): Dropout(p=0.0, inplace=False)
            (proj): Linear(in_features=512, out_features=512, bias=True)
            (proj_drop): Dropout(p=0.0, inplace=False)
            (softmax): Softmax(dim=-1)
          )
          (drop_path): DropPath()
          (norm2): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
          (mlp): Mlp(
            (fc1): Linear(in_features=512, out_features=2048, bias=True)
            (act): GELU()
            (fc2): Linear(in_features=2048, out_features=512, bias=True)
            (drop): Dropout(p=0.0, inplace=False)
          )
        )
        (14): SwinTransformerBlock(
          dim=512, input_resolution=(14, 14), num_heads=16, window_size=7, shift_size=0, mlp_ratio=4.0
          (norm1): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
          (attn): WindowAttention(
            dim=512, window_size=(7, 7), num_heads=16
            (qkv): Linear(in_features=512, out_features=1536, bias=True)
            (attn_drop): Dropout(p=0.0, inplace=False)
            (proj): Linear(in_features=512, out_features=512, bias=True)
            (proj_drop): Dropout(p=0.0, inplace=False)
            (softmax): Softmax(dim=-1)
          )
          (drop_path): DropPath()
          (norm2): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
          (mlp): Mlp(
            (fc1): Linear(in_features=512, out_features=2048, bias=True)
            (act): GELU()
            (fc2): Linear(in_features=2048, out_features=512, bias=True)
            (drop): Dropout(p=0.0, inplace=False)
          )
        )
        (15): SwinTransformerBlock(
          dim=512, input_resolution=(14, 14), num_heads=16, window_size=7, shift_size=3, mlp_ratio=4.0
          (norm1): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
          (attn): WindowAttention(
            dim=512, window_size=(7, 7), num_heads=16
            (qkv): Linear(in_features=512, out_features=1536, bias=True)
            (attn_drop): Dropout(p=0.0, inplace=False)
            (proj): Linear(in_features=512, out_features=512, bias=True)
            (proj_drop): Dropout(p=0.0, inplace=False)
            (softmax): Softmax(dim=-1)
          )
          (drop_path): DropPath()
          (norm2): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
          (mlp): Mlp(
            (fc1): Linear(in_features=512, out_features=2048, bias=True)
            (act): GELU()
            (fc2): Linear(in_features=2048, out_features=512, bias=True)
            (drop): Dropout(p=0.0, inplace=False)
          )
        )
        (16): SwinTransformerBlock(
          dim=512, input_resolution=(14, 14), num_heads=16, window_size=7, shift_size=0, mlp_ratio=4.0
          (norm1): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
          (attn): WindowAttention(
            dim=512, window_size=(7, 7), num_heads=16
            (qkv): Linear(in_features=512, out_features=1536, bias=True)
            (attn_drop): Dropout(p=0.0, inplace=False)
            (proj): Linear(in_features=512, out_features=512, bias=True)
            (proj_drop): Dropout(p=0.0, inplace=False)
            (softmax): Softmax(dim=-1)
          )
          (drop_path): DropPath()
          (norm2): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
          (mlp): Mlp(
            (fc1): Linear(in_features=512, out_features=2048, bias=True)
            (act): GELU()
            (fc2): Linear(in_features=2048, out_features=512, bias=True)
            (drop): Dropout(p=0.0, inplace=False)
          )
        )
        (17): SwinTransformerBlock(
          dim=512, input_resolution=(14, 14), num_heads=16, window_size=7, shift_size=3, mlp_ratio=4.0
          (norm1): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
          (attn): WindowAttention(
            dim=512, window_size=(7, 7), num_heads=16
            (qkv): Linear(in_features=512, out_features=1536, bias=True)
            (attn_drop): Dropout(p=0.0, inplace=False)
            (proj): Linear(in_features=512, out_features=512, bias=True)
            (proj_drop): Dropout(p=0.0, inplace=False)
            (softmax): Softmax(dim=-1)
          )
          (drop_path): DropPath()
          (norm2): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
          (mlp): Mlp(
            (fc1): Linear(in_features=512, out_features=2048, bias=True)
            (act): GELU()
            (fc2): Linear(in_features=2048, out_features=512, bias=True)
            (drop): Dropout(p=0.0, inplace=False)
          )
        )
      )
      (downsample): PatchMerging(
        input_resolution=(14, 14), dim=512
        (reduction): Linear(in_features=2048, out_features=1024, bias=False)
        (norm): LayerNorm((2048,), eps=1e-05, elementwise_affine=True)
      )
    )
    (3): BasicLayer(
      dim=1024, input_resolution=(7, 7), depth=2
      (blocks): ModuleList(
        (0): SwinTransformerBlock(
          dim=1024, input_resolution=(7, 7), num_heads=32, window_size=7, shift_size=0, mlp_ratio=4.0
          (norm1): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
          (attn): WindowAttention(
            dim=1024, window_size=(7, 7), num_heads=32
            (qkv): Linear(in_features=1024, out_features=3072, bias=True)
            (attn_drop): Dropout(p=0.0, inplace=False)
            (proj): Linear(in_features=1024, out_features=1024, bias=True)
            (proj_drop): Dropout(p=0.0, inplace=False)
            (softmax): Softmax(dim=-1)
          )
          (drop_path): DropPath()
          (norm2): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
          (mlp): Mlp(
            (fc1): Linear(in_features=1024, out_features=4096, bias=True)
            (act): GELU()
            (fc2): Linear(in_features=4096, out_features=1024, bias=True)
            (drop): Dropout(p=0.0, inplace=False)
          )
        )
        (1): SwinTransformerBlock(
          dim=1024, input_resolution=(7, 7), num_heads=32, window_size=7, shift_size=0, mlp_ratio=4.0
          (norm1): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
          (attn): WindowAttention(
            dim=1024, window_size=(7, 7), num_heads=32
            (qkv): Linear(in_features=1024, out_features=3072, bias=True)
            (attn_drop): Dropout(p=0.0, inplace=False)
            (proj): Linear(in_features=1024, out_features=1024, bias=True)
            (proj_drop): Dropout(p=0.0, inplace=False)
            (softmax): Softmax(dim=-1)
          )
          (drop_path): DropPath()
          (norm2): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
          (mlp): Mlp(
            (fc1): Linear(in_features=1024, out_features=4096, bias=True)
            (act): GELU()
            (fc2): Linear(in_features=4096, out_features=1024, bias=True)
            (drop): Dropout(p=0.0, inplace=False)
          )
        )
      )
    )
  )
  (norm): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
  (avgpool): AdaptiveAvgPool1d(output_size=1)
  (head): Linear(in_features=1024, out_features=100, bias=True)
)
[2022-11-03 10:20:18 swin_base_patch4_window7_224](main.py 96): INFO number of params: 86845724





input （N,C,Lin）or (N,Lin)
output （N,C,Lout）or (N,Lout)
self.avgpool = nn.AdaptiveAvgPool1d(1)  自适应输出一个维度



self.num_features = int(embed_dim * 2 ** (self.num_layers - 1)) 随着层数dim维度增加 embed_dim * 2 ** （4-1）=1024



Input: :math:`(*, H_{in})` 
Output: :math:`(*, H_{out})`
self.head = nn.Linear(self.num_features, num_classes) bath的数量不需要关注





self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
self.relative_position_bias_table = nn.Parameter(torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH















