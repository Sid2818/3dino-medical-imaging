# 3dino-medical-imaging

### dataset:
coca coronary calciuma nd chest CTs (standford datasets)

### backbone: 
3dino model from AICONSlab
https://huggingface.co/AICONSlab/3DINO-ViT/tree/main

### decoder: 
ViTAdapterUNETR decoder from AICONSlab github

### low gpu VRAM:
num workers = 0,
batch size = 2

### high GPU VRAM:
num workers = 4 or 8,
batch size = 4

### pipeline: 
dataset -> data_preprocess -> fetch_dataset -> 3dino_semseg_new
