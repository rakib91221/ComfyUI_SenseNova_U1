 # !/usr/bin/env python
# -*- coding: UTF-8 -*-

import numpy as np
import torch
import os
from comfy_api.latest import  io
import folder_paths

from .node_utils import  tensor2pillist,clear_comfyui_cache
from .SenseNova.examples.editing.inference import load_sensenova_model,infer_sensenova_edit
from .SenseNova.examples.t2i.inference import infer_sensenova_t2i
from .SenseNova.examples.interleave.inference import infer_sensenova_interleave
from .SenseNova.examples.vqa.inference import infer_sensenova_vqa


device = torch.device(
    "cuda:0") if torch.cuda.is_available() else torch.device(
    "mps") if torch.backends.mps.is_available() else torch.device(
    "cpu")

MAX_SEED = np.iinfo(np.int32).max
node_cr_path = os.path.dirname(os.path.abspath(__file__))
weigths_gguf_current_path = os.path.join(folder_paths.models_dir, "gguf")
if not os.path.exists(weigths_gguf_current_path):
    os.makedirs(weigths_gguf_current_path)
folder_paths.add_model_folder_path("gguf", weigths_gguf_current_path) #  gguf dir

class SenseNova_SM_Model(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SenseNova_SM_Model",
            display_name="SenseNova_SM_Model",
            category="SenseNova",
            inputs=[
                io.Combo.Input("diffusion_models",options= ["none"] + folder_paths.get_filename_list("diffusion_models")),
                io.Combo.Input("gguf",options= ["none"] + folder_paths.get_filename_list("gguf")),
                io.Combo.Input("attn_backend",options= ["auto", "flash", "sdpa"]),
            ],
            outputs=[
                io.Model.Output(display_name="model"),
                ],
            )
    @classmethod
    def execute(cls, diffusion_models,gguf,attn_backend) -> io.NodeOutput:
        clear_comfyui_cache()
        dit_path=folder_paths.get_full_path("diffusion_models",diffusion_models) if diffusion_models != "none" else None
        gguf_path=folder_paths.get_full_path("gguf",gguf) if gguf != "none" else None
        model_path=dit_path or gguf_path
        model=load_sensenova_model(model_path,device,os.path.join(node_cr_path,"SenseNova-U1-8B-MoT-SFT"),attn_backend)
        return io.NodeOutput(model)
    

class SenseNova_SM_Sampler(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SenseNova_SM_Sampler",
            display_name="SenseNova_SM_Sampler",
            category="SenseNova",
            inputs=[
                io.Model.Input("model"),
                io.Combo.Input("img_mode",options= ["edit",  "interleave", "vqa"]),
                io.String.Input("prompt",default="a photo of a cat",multiline=True),
                io.Int.Input("seed", default=0, min=0, max=MAX_SEED),
                io.Int.Input("steps", default=8, min=1, max=10000, step=1),
                io.Combo.Input("target_pixels",options= [[2048, 2048], [2720, 1536], [1536, 2720], [2496, 1664],[1664, 2496],[2368, 1760],[1760, 2368],[1440, 2880],[2880, 1440],[1152, 3456],[3456, 1152]]),
                io.Float.Input("cfg", default=1.0, min=0.0, max=10.0, step=0.1, round=0.01,),
                io.Float.Input("img_cfg", default=1.0, min=0.0, max=100.0, step=0.1, round=0.01,),
                io.Float.Input("timestep_shift", default=3.0, min=-1.0, max=10.0, step=0.1, ),
                io.Int.Input("batch_size", default=1, min=1, max=64,step=1),
                io.Int.Input("prefetch_count", default=1, min=0, max=64,step=1),
                io.Combo.Input("cfg_norm",options= ["none", "global", "channel"]),
                io.Boolean.Input("enhance", default=False),
                io.Boolean.Input("think_mode", default=False),
                io.Boolean.Input("do_sample", default=True),
                io.Int.Input("max_new_tokens", default=1024, min=256, max=10241024,step=1),
                io.Float.Input("temperature", default=0.7, min=0.0, max=1.0, step=0.1,),
                io.Float.Input("top_p", default=0.9, min=0.0, max=1.0, step=0.1,),
                io.Int.Input("top_k", default=0, min=0, max=1024,step=1),
                io.Float.Input("repetition_penalty", default=0.0, min=0.0, max=10.0, step=0.1,),
                io.Image.Input("image",optional=True),
            ],
            outputs=[
                io.Image.Output(display_name="image"),
                io.String.Output(display_name="text"),
            ],
        )
    
    @classmethod
    def execute(cls, model, img_mode,prompt,seed, steps, target_pixels,cfg,img_cfg,timestep_shift,batch_size,prefetch_count,cfg_norm,enhance,
                think_mode,do_sample,max_new_tokens,temperature,top_p,top_k,repetition_penalty,image=None) -> io.NodeOutput:
        clear_comfyui_cache()
        top_k=None if top_k==0 else top_k
        repetition_penalty=repetition_penalty if repetition_penalty>0.0 else None
        cfg_interval=[0.0,1.0]
        width,height=target_pixels
        images=tensor2pillist(image) if image is not None else None
        
        if prefetch_count==0:
            model.model.to(device)
            prefetch_count=None
        
        if images is not None:
            print(f"infer_mode is : {img_mode}")
            if "edit"==img_mode:
                image,text=infer_sensenova_edit(model,prompt,cfg,cfg_norm,steps,batch_size,timestep_shift,img_cfg,cfg_interval,width,height,images,target_pixels,seed,prefetch_count,think_mode)
            elif "vqa"==img_mode:
                image=torch.zeros((1,height, width,3))
                text=infer_sensenova_vqa(model,prompt,images[0],max_new_tokens,do_sample,temperature,top_p,top_k,repetition_penalty,prefetch_count)
            else:
                text,image=infer_sensenova_interleave(model,prompt,cfg,steps,timestep_shift,img_cfg,images,cfg_interval,width,height,think_mode,seed,prefetch_count)
        else:
            if "interleave"==img_mode:
                print(f"infer_mode is : interleave without image")
                text,image=infer_sensenova_interleave(model,prompt,cfg,steps,timestep_shift,img_cfg,images,cfg_interval,width,height,think_mode,seed,prefetch_count)
            else:
                print(f"infer_mode is : t2i")
                text,image=infer_sensenova_t2i(model,prompt,cfg,cfg_norm,steps,batch_size,timestep_shift,cfg_interval,width,height,seed,prefetch_count,think_mode,enhance,)

        if  prefetch_count is None:
            model.model.to(torch.device("cpu"))

        return io.NodeOutput(image,text)
