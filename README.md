# Cyst Segmentation Training

Repo nay dung de train/evaluate segmentation cyst voi ca model 2D va 3D.
Config goc la `config/cyst.yaml`; config rieng cho tung model nam trong
`config/2D_model` va `config/3D_model`.

## 1. Cai moi truong

Repo hien tai duoc chinh cho Python 3.12.

```bash
python -m venv seg
source seg/bin/activate
pip install -r requirements.txt
```

`requirements.txt` da them cac thu vien can cho PyTorch, MONAI, nnUNet-style
model, DeepLab/DeepLab++, Swin-UNet, metric surface va visualize.

## 2. Data split

Mac dinh data root la `data/`.

```text
data/train_new.txt
data/val_new.txt
data/all_train.txt
data/test.txt
```

Moi file split can co column:

```csv
image_path,mask_path
```

Ghi chu:
- `train_new.txt` va `val_new.txt` dung khi `k_fold.enabled: false`.
- `all_train.txt` dung de chia k-fold train/val.
- `test.txt` la test doc lap, khong dua vao train/val trong k-fold.

## 3. Chon model

Nen train bang config rieng:

```text
config/2D_model/*.yaml
config/3D_model/*.yaml
```

Script tuong ung:

```text
scripts/2D_model/*.sh
scripts/3D_model/*.sh
```

Vi du:

```bash
bash scripts/2D_model/train_unet.sh
bash scripts/2D_model/train_deeplab.sh
bash scripts/2D_model/train_deeplab_plus_plus.sh
bash scripts/2D_model/train_swin_unet.sh
bash scripts/3D_model/train_unet3d.sh
bash scripts/3D_model/train_vnet.sh
```

Hoac chay truc tiep:

```bash
python main.py --config config/2D_model/unet.yaml
```

## 4. Model dang ho tro

2D:

```text
unet, resunet, vnet, unetr, attention_unet/att_unet, r2unet,
unet_plus_plus/unet_plus_pluss, unet_resnet152,
unet_3_plus/unet3plus, unet3plus_hybrid_cgm,
swin_unet, deeplab, deeplab_plus_plus, nnunet/nnunet2d
```

3D:

```text
unet3d/unet, vnet, unetr, nnunet/nnunet3d, mobi_style_3d
```

Backbone/pretrain duoc comment trong tung file config. Cac backbone ImageNet
pretrained se duoc cache tai:

```text
pretrain/hub/checkpoints
```

Neu bat pretrain ma checkpoint chua co, code se tai ve cache. Neu da co thi
dung lai file da tai. Neu tai/load that bai, model se fallback train tu dau.

## 5. Train 2D va danh gia nhu 3D

Voi model 2D, mac dinh:

```yaml
evaluation:
  evaluate_2d_as_volume: true
```

Nghia la khi evaluate, model 2D se predict tat ca slice cua moi volume, stack
lai thanh volume 3D, roi moi tinh Dice/IoU/HD95/ASD. Vi vay metric 2D va 3D
duoc so sanh tren cung kieu full-volume evaluation.

`slice_2d` chi quyet dinh cach tao sample 2D de train:

```yaml
slice_2d:
  num_slices: 1
  axis: 2
  position: center
  sampling_strategy: center
  samples_per_volume: 1
```

Ghi chu nhanh:
- `num_slices=1`: input 1 channel, predict 1 slice.
- `num_slices=3`: input 3 channel `[z-1,z,z+1]`, predict mask slice giua.
- `sampling_strategy=center`: lay slice giua.
- `uniform`: lay `samples_per_volume` slice rai deu.
- `random`: lay slice ngau nhien co seed.
- `all`: lay tat ca slice.

## 6. K-fold

Bat k-fold trong config:

```yaml
k_fold:
  enabled: true
  source_list: all_train.txt
  num_folds: 5
  shuffle: true
  seed: 42
```

Code se chia `all_train.txt` thanh 5 fold. Moi fold train tren 80% va validate
tren 20%. `test.txt` luon doc lap va duoc evaluate cuoi moi fold.

Output k-fold gom:

```text
metrics_kfold.csv
metrics_kfold_summary.csv
fold_01/
fold_02/
...
```

`metrics_kfold_summary.csv` co mean/std de bao cao ket qua.

## 7. Output

Output co dang:

```text
outputs/<model>/<slice_selection>/<real_encoder>_epoch<epochs>/
```

Vi du:

```text
outputs/unet/pos_center_sam_center_spv1_sl1_ax2/unet_encoder_epoch60/
```

Thu muc quan trong:

```text
checkpoint/best.pth
metrics.csv
curve.pdf
predictions/<train|val|test>/
visualize/<train|val|test>/
```

Trong moi split visualize/predictions co:

```text
panel/                  PDF gom image, GT, predict, logit
image/                  slice image PNG
gt/                     ground truth PNG
predict/                prediction PNG
logit/                  foreground logit map PNG
feature_mean/           feature map trung binh theo channel
encoder_activation/     activation heatmap tung encoder stage
skip_connection/        skip feature truoc decoder
gradcam_bottleneck/     Grad-CAM bottleneck
gradcam_decoder/        Grad-CAM decoder
visualization_index.csv case_id/source/slice da visualize
```

Voi config mac dinh, `visualize/<split>` la thu muc chinh de so sanh dac trung:
moi source/khu vuc chi co `samples_per_source: 1` case va moi case chi ve
`1` slice duy nhat. Cac folder `feature_mean`, `encoder_activation`,
`skip_connection`, `gradcam_bottleneck`, `gradcam_decoder`, `logit` deu duoc ve
tren cung sample va cung slice do.

## 8. Visualize dong bo giua cac model

Config mac dinh da de:

```yaml
visualization:
  seed: 42
  selection: per_source
  samples_per_source: 1
  slice_axis: 2
  slice_position: label_foreground
```

Y nghia:
- Moi source/khu vuc lay 1 sample.
- Sample duoc chon on dinh theo `case_id` va `seed`.
- Slice visualize la slice co GT foreground nhieu nhat.
- Khong phu thuoc prediction cua model, nen cac model se cung sample/cung slice.
- `visualize/<split>/visualization_index.csv` la file de kiem tra model nao
  dang ve case nao, source nao, slice nao.

Voi model 2D, visualization cung dung full-volume dataset khi
`evaluate_2d_as_volume: true`, nen slice visualize dong bo voi 3D hon.

## 9. Visualize lai tu weight da train

Dung khi ban da train nhieu model va muon tao lai visualize dong bo cho tat ca:

```bash
bash scripts/visualize_from_weight.sh ALL outputs/visualize_synced
```

Script se scan:

```text
outputs/**/checkpoint/best.pth
```

Sau do load weight, dung config luu trong checkpoint, nhung ep cac tham so
visualize dong bo:

```text
VISUAL_SELECTION=per_source
SAMPLES_PER_SOURCE=1
VISUAL_SEED=42
SLICE_POSITION=label_foreground
```

Output moi se nam trong:

```text
outputs/visualize_synced/...
```

Chay lai 1 checkpoint:

```bash
bash scripts/visualize_from_weight.sh \
  config/2D_model/unet.yaml \
  outputs/unet/pos_center_sam_center_spv1_sl1_ax2/unet_encoder_epoch60/checkpoint/best.pth \
  outputs/visualize_synced/unet \
  label_foreground
```

Neu checkpoint thuoc k-fold va nam trong folder `fold_03`, script `ALL` se tu
nhan fold. Khi chay 1 checkpoint rieng co the them tham so fold o cuoi:

```bash
bash scripts/visualize_from_weight.sh CONFIG CHECKPOINT OUTPUT_DIR label_foreground 3
```

## 10. GPU

Config GPU:

```yaml
gpu:
  use_cuda: true
  ids: "0, 1"
  multi_gpu: true
```

Neu `multi_gpu: true`, code dung DataParallel. Batch tong duoc chia tren cac GPU,
gradient duoc gom lai roi update 1 lan moi step.

Voi 3D, neu OOM hay giam:

```yaml
training:
  batch_size_3d: 4
  image_size: [256, 256, 64]
```

## 11. Loss ablation L1-L6

Loss moi nam trong package `losses/` va van tuong thich config cu:

```yaml
training:
  loss: ce_dice
```

De chay ablation tren proposal `mobi_style_3d_v2`:

```bash
bash scripts/loss_ablation/train_L1_mobi_style_3d_v2.sh  # Dice + BCE
bash scripts/loss_ablation/train_L2_mobi_style_3d_v2.sh  # Dice + Focal
bash scripts/loss_ablation/train_L3_mobi_style_3d_v2.sh  # Dice + Focal Tversky
bash scripts/loss_ablation/train_L4_mobi_style_3d_v2.sh  # L3 + Boundary
bash scripts/loss_ablation/train_L5_mobi_style_3d_v2.sh  # L3 + Encoder Attention
bash scripts/loss_ablation/train_L6_mobi_style_3d_v2.sh  # L3 + Boundary + Attention
```

Output tung loss duoc tach rieng:

```text
outputs/loss_ablation/L1_dice_bce/...
outputs/loss_ablation/L6_proposed/...
```

Train CSV se co them cac cot component neu loss co dung:
`train_loss_dice`, `train_loss_focal_tversky`, `train_loss_boundary`,
`train_loss_attention`, va cac cot `val_loss_*`.

Test nhanh loss:

```bash
python scripts/test_losses.py
```

## 12. Lenh nhanh

Train UNet 2D:

```bash
bash scripts/2D_model/train_unet.sh
```

Train DeepLab++ 2D:

```bash
bash scripts/2D_model/train_deeplab_plus_plus.sh
```

Train VNet 3D:

```bash
bash scripts/3D_model/train_vnet.sh
```

Train proposal Mobile-style 2D-3D U-Net:

```bash
bash scripts/Proposal/train_mobi_style_3d.sh
```

Train proposal v3 giu nguyen depth goc, chi resize H/W:

```bash
bash scripts/Proposal/train_mobi_style_3d_v3.sh
```

Visualize lai tat ca model da train:

```bash
bash scripts/visualize_from_weight.sh ALL outputs/visualize_synced
```

Chon GPU khi visualize lai:

```bash
GPU_IDS=4 bash scripts/visualize_from_weight.sh ALL outputs/visualize_synced
```

Neu muon expose nhieu GPU:

```bash
GPU_IDS="4,5" MULTI_GPU=true bash scripts/visualize_from_weight.sh ALL outputs/visualize_synced
```

Ghi chu: visualize hien chay tung sample/slice nen thuong `GPU_IDS=4` la hop ly
nhat. Multi-GPU chi huu ich neu model/code path co batch lon; neu khong thi nen
chay nhieu job song song voi cac `GPU_IDS` khac nhau.
