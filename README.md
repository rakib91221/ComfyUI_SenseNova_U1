# ComfyUI_SenseNova_U1
[SenseNova-U1](https://github.com/OpenSenseNova/SenseNova-U1): Unifying Multimodal Understanding and Generation with NEO-Unify Architecture

# Update
* fix interleave some bugs ,add interleave max images number, 修复bug，交叉模式生成图片数量可以选入参，注意因为kv缓存的原因，越大越占用显存
* support 8 steps lora now  支持 8步lora
* Test it use 8G Vram 36G Ram ,确保内存（不是显存）大于36G
* If Vram >16G make prefetch_count =0, 显存大于16G时，设置swap（prefetch_count）数值为0以关闭层交换（使用Q6 gguf时）

1.Installation  
-----
  In the ./ComfyUI/custom_nodes directory, run the following:   
```
git clone https://github.com/smthemex/ComfyUI_SenseNova_U1
```
2.requirements  
----
```
pip install -r requirements.txt
```
If some modules missing, please pip install   #ultralytics yolov8

3.checkpoints 
----
[links](https://huggingface.co/smthem/SenseNova-U1-8B-MoT-Merger-gguf)  
[lora](https://huggingface.co/sensenova/SenseNova-U1-8B-MoT-LoRAs)
[夸克网盘](https://pan.quark.cn/s/8180628d73c5)
    
```
├── ComfyUI/models/gguf/
|     ├── SenseNova-U1-8B-MoT-8step-Q6_K.gguf # optional 可选
├── ComfyUI/models/diffusion_models/
|     ├── SenseNova-U1-8B-MoT-8step-merge_bf16.safetensors # optional 可选
├── ComfyUI/models/loras/
|     ├── SenseNova-U1-8B-MoT-LoRA-8step-V1.0.safetensors # optional 可选

```

4. Example
----
![](https://github.com/smthemex/ComfyUI_SenseNova_U1/blob/main/example_workflows/example_in.png)
![](https://github.com/smthemex/ComfyUI_SenseNova_U1/blob/main/example_workflows/example_lora.png)
![](https://github.com/smthemex/ComfyUI_SenseNova_U1/blob/main/example_workflows/example_edit.png)
![](https://github.com/smthemex/ComfyUI_SenseNova_U1/blob/main/example_workflows/example_ti2i.png)
![](https://github.com/smthemex/ComfyUI_SenseNova_U1/blob/main/example_workflows/example_t2i.png)

Citation
-----
