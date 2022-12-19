# AI_CUP_2022_Farming_TEAM_2154

**在Colab上掛載Google Drive**

``` 
from google.colab import drive
drive.mount("/content/drive")
```
**進入到文件所在的目錄**

``` python
import os
path = "/content/drive/My Drive/AI_CUP2022_Farming_TEAM_2154"
os.chdir(path)
os.listdir(path)
```

**需安裝套件:** 

`pip install timm==0.6.11`

``` python
pip install timm==0.6.11
```



**1.執行輸出預測答案**

1-1 先將 private_test.zip 和 public_test.zip 解壓縮至
(./total_test/)

**File Structure**

    ├── total_test
	    	├── 0
		│     ├──*.jpg
		│     │
		│     │  ...
		│     │
		│     ├──*.jpg
		│ 
		│ ...
		│ 
		└── f
		      ├──*.jpg
		      │
		      │  ...
		      │
		      ├──*.jpg



1-2 接著下載6份權重檔，放在(./weight/)

https://drive.google.com/drive/folders/1AIiLbCOiZOGQZLWFa4S8b8YKVq4Bkycl?usp=share_link

https://drive.google.com/drive/folders/1Ljl4Alqfo34yKMstZmNgmggCaE08o5m0?usp=share_link

1-3 執行 Final_Test_ensemble.py 即可輸出答案，答案會儲存在(./Ans)

`!python Final_Test_ensemble.py`
``` python
!python Final_Test_ensemble.py
```




**2.如何執行訓練**

將 train_data.zip 解壓縮至
(./Crop33/data/)

**File Structure**

    ├──./Crop33/data/
	    	├── asparagus
		│     ├──*.jpg
		│     │
		│     │  ...
		│     │
		│     ├──*.jpg
		│ 
		│ ...
		│ 
		└── cauliflower
		      ├──*.jpg
		      │
		      │  ...
		      │
		      ├──*.jpg


**進入pytorch-image-models 目錄下**
``` python
%cd pytorch-image-models
```
**訓練指令**

--experiment 儲存訓練實驗的檔案夾名稱

訓練 swinv2-A
``` python
!CUDA_LAUNCH_BLOCKING=1 python train.py ../Crops33/data --model swinv2_large_window12to24_192to384_22kft1k --pretrained --num-classes 33 -b 12 -vb 12 --opt adamw --weight-decay 0.01 --layer-decay 0.65 --sched cosine --lr 0.0001 --lr-cycle-limit 1 --warmup-lr 1e-5 --min-lr 1e-5 --epochs 30 --warmup-epochs 5 --color-jitter 0.5 --reprob 0.5 --scale 0.4 1.0 --train-interpolation bicubic --drop-path 0.1 -j 20 --save-images --output output --experiment swinv2_large_window12to24_192to384_22kft1k_newA_mean_std --train_txt ./datasets_txt/A_train.txt --val_txt ./datasets_txt/A_val.txt --mean 0.45925 0.48785 0.42035 --std 0.25080 0.24715 0.29270
```

訓練 swinv2-B
``` python
!CUDA_LAUNCH_BLOCKING=1 python train.py ../Crops33/data --model swinv2_large_window12to24_192to384_22kft1k --pretrained --num-classes 33 -b 12 -vb 12 --opt adamw --weight-decay 0.01 --layer-decay 0.65 --sched cosine --lr 0.0001 --lr-cycle-limit 1 --warmup-lr 1e-5 --min-lr 1e-5 --epochs 30 --warmup-epochs 5 --color-jitter 0.5 --reprob 0.5 --scale 0.4 1.0 --train-interpolation bicubic --drop-path 0.1 -j 20 --save-images --output output --experiment swinv2_large_window12to24_192to384_22kft1k_newB_mean_std --train_txt ./datasets_txt/B_train.txt --val_txt ./datasets_txt/B_val.txt --mean 0.45925 0.48785 0.42035 --std 0.25080 0.24715 0.29270
```

訓練 beit-A
``` python
!CUDA_LAUNCH_BLOCKING=1 python train.py ../Crops33/data --model beit_large_patch16_384 --pretrained --num-classes 33 -b 24 -vb 24 --opt adamw --weight-decay 0.01 --layer-decay 0.65 --sched cosine --lr 0.0001 --lr-cycle-limit 1 --warmup-lr 1e-5 --min-lr 1e-5 --epochs 300 --warmup-epochs 5 --color-jitter 0.5 --reprob 0.5 --scale 0.4 1.0 --train-interpolation bicubic --drop-path 0.1 -j 20 --save-images --output output --experiment beit_large_patch16_384_newA_baseline --train_txt ./datasets_txt/A_train.txt --val_txt ./datasets_txt/A_val.txt --crop-pct 1.0
```

訓練 beit-B
``` python
!CUDA_LAUNCH_BLOCKING=1 python train.py ../Crops33/data --model beit_large_patch16_384 --pretrained --num-classes 33 -b 24 -vb 24 --opt adamw --weight-decay 0.01 --layer-decay 0.65 --sched cosine --lr 0.0001 --lr-cycle-limit 1 --warmup-lr 1e-5 --min-lr 1e-5 --epochs 30 --warmup-epochs 5 --color-jitter 0.5 --reprob 0.5 --scale 0.4 1.0 --train-interpolation bicubic --drop-path 0.1 -j 20 --save-images --output output --experiment beit_large_patch16_384_newB_baseline --train_txt ./datasets_txt/B_train.txt --val_txt ./datasets_txt/B_val.txt --crop-pct 1.0
```

訓練 swinv2-A-KD
``` python
!CUDA_LAUNCH_BLOCKING=1 python train_distillation.py ../Crops33/data --model swinv2_large_window12to24_192to384_22kft1k --pretrained --num-classes 33 -b 6 -vb 6 --opt adamw --weight-decay 0.01 --layer-decay 0.65 --sched cosine --lr 0.0001 --lr-cycle-limit 1 --warmup-lr 1e-5 --min-lr 1e-5 --epochs 30 --warmup-epochs 5 --color-jitter 0.5 --reprob 0.5 --scale 0.4 1.0 --train-interpolation bicubic --drop-path 0.1 -j 20 --save-images --output output --experiment swinv2_large_window12to24_192to384_22kft1k_newA_mean_std_distillation --train_txt ./datasets_txt/A_train.txt --val_txt ./datasets_txt/A_val.txt --mean 0.45925 0.48785 0.42035 --std 0.25080 0.24715 0.29270
```

訓練 swinv2-B-KD
``` python
!CUDA_LAUNCH_BLOCKING=1 python train_distillation.py ../Crops33/data --model swinv2_large_window12to24_192to384_22kft1k --pretrained --num-classes 33 -b 6 -vb 6 --opt adamw --weight-decay 0.01 --layer-decay 0.65 --sched cosine --lr 0.0001 --lr-cycle-limit 1 --warmup-lr 1e-5 --min-lr 1e-5 --epochs 30 --warmup-epochs 5 --color-jitter 0.5 --reprob 0.5 --scale 0.4 1.0 --train-interpolation bicubic --drop-path 0.1 -j 20 --save-images --output output --experiment swinv2_large_window12to24_192to384_22kft1k_newB_mean_std_distillation --train_txt ./datasets_txt/B_train.txt --val_txt ./datasets_txt/B_val.txt --mean 0.45925 0.48785 0.42035 --std 0.25080 0.24715 0.29270
```

**備註-指定單張GPU**

`CUDA_VISIBLE_DEVICES=1 bash ./distributed_train.sh 1`

**備註-指定多張GPU**

`CUDA_VISIBLE_DEVICES=2,3 bash ./distributed_train.sh 2`

**備註-呼叫標籤引數**

```python
from timm.data.parsers.parser_image_folder import find_images_and_targets
samples, class_to_idx=find_images_and_targets(folder='./',class_to_idx=None)
```
