#!/usr/bin/env python3

import torch 
import logging 
import sys 
import os 
import inspect 
from tqdm import tqdm 
import torch .nn as nn 
from typing import Tuple ,Dict ,Any ,Optional 


def classify_checkpoint_stage (ckpt :Dict [str ,Any ],model :nn .Module ,cfg :Dict [str ,Any ])->str :
    """Infer the training stage from checkpoint contents."""
    stage =ckpt .get ("stage")or ckpt .get ("config",{}).get ("_current_stage")
    if stage :
        return stage 

    def _tensor_id_name_map (mod :nn .Module )->Dict [int ,str ]:
        return {id (p ):n for n ,p in mod .named_parameters ()}

    id2name =_tensor_id_name_map (model )
    try :
        param_ids =ckpt ["optimizer"]["param_groups"][0 ]["params"]
    except Exception :
        param_ids =[]

    names =[id2name .get (pid ,"")for pid in param_ids ]
    has_ppo ="ppo_optimizer"in ckpt 

    if names and all ("raw_c"in n or "raw_r"in n for n in names )and not has_ppo :
        return "curvature_calibration"
    if has_ppo and any ("thought_generator"in n for n in names ):
        return "reasoning_warmup"
    if any (n .startswith ("experts.")for n in names )and not has_ppo :
        return "branch_pretrain"
    if any ("gate"in n or "combiner"in n for n in names )and not has_ppo :
        return "gate_train"
    if has_ppo :
        return "joint_finetune"
    return cfg .get ("_current_stage","unknown")


def load_and_validate_checkpoint (
model :nn .Module ,
checkpoint_path :str ,
config :dict ,
expected_stage :str ,
)->Tuple [int ,int ,int ]:
    """Load checkpoint ensuring stage and tensor shapes match."""
    if not os .path .exists (checkpoint_path ):
        raise FileNotFoundError (checkpoint_path )

    ckpt =torch .load (checkpoint_path ,map_location ="cpu",weights_only =False )
    stage =classify_checkpoint_stage (ckpt ,model ,{"_current_stage":expected_stage ,**config })

    if stage !=expected_stage :
        raise ValueError (f"Checkpoint stage {stage } does not match expected {expected_stage }")

    state_dict =ckpt .get ("model_state_dict",ckpt .get ("state_dict",ckpt ))

    mismatched =[]
    model_state =model .state_dict ()
    for name ,tensor in state_dict .items ():
        if name in model_state and model_state [name ].shape !=tensor .shape :
            mismatched .append ((name ,tuple (tensor .shape ),tuple (model_state [name ].shape )))

    if mismatched :
        details =", ".join (f"{n }: {c } vs {m }"for n ,c ,m in mismatched )
        raise ValueError (f"Parameter shape mismatch: {details }")

    load_result =model .load_state_dict (state_dict ,strict =False )
    if load_result .missing_keys :
        logging .warning (f"Missing keys in state_dict: {load_result .missing_keys }")
    if load_result .unexpected_keys :
        logging .warning (f"Unexpected keys in state_dict: {load_result .unexpected_keys }")
    return ckpt .get ("epoch",0 ),ckpt .get ("batch",0 ),ckpt .get ("global_step",0 )


def train_epoch_with_resume (
original_train_func ,
model ,
reward_model ,
data_loader ,
optimizer ,
ppo_optimizer ,
criterion ,
device ,
epoch ,
config ,
global_step =0 ,
start_batch =0 ,
lr_scheduler =None ,
**extra_kwargs ,
):
    """Resume-aware wrapper that skips to *start_batch* before calling the original function."""

    print (
    f"🔄 RESUME MODE: Epoch {epoch }, Global Step {global_step }, Start Batch {start_batch }"
    )

    from itertools import islice 

    max_batches =config .get ('training',{}).get ('max_steps')

    class _ResumeLoader :
        def __init__ (self ,loader ,skip :int =0 ,limit :Optional [int ]=None ):
            self .loader =loader 
            self .skip =skip 
            self .limit =limit 

        def __iter__ (self ):
            it =iter (self .loader )
            if self .skip :
                it =islice (it ,self .skip ,None )
            if self .limit is not None :
                it =islice (it ,0 ,self .limit )
            return it 

        def __len__ (self ):
            try :
                n =len (self .loader )
                n =max (0 ,n -self .skip )
            except Exception :
                n =self .limit if self .limit is not None else 0 
            if self .limit is not None :
                return min (n ,self .limit )
            return n 

    if start_batch >0 or max_batches is not None :
        data_loader =_ResumeLoader (data_loader ,start_batch ,max_batches )

    return original_train_func (
    model ,
    reward_model ,
    data_loader ,
    optimizer ,
    ppo_optimizer ,
    criterion ,
    device ,
    epoch ,
    config ,
    global_step ,
    lr_scheduler =lr_scheduler ,
    **extra_kwargs ,
    )


def patch_trainer_for_resume (trainer ,save_steps :int =25 )->None :
    """Monkey patch *trainer* so orchestrate_training_stage can resume and save checkpoints frequently."""
    if getattr (trainer ,"_resume_patch_applied",False ):
        return 

    original =trainer .train_epoch 


    try :
        src =inspect .getsource (original )
        if "batch_idx % 5000 == 0"in src :
            patched_src =src .replace ("batch_idx % 5000 == 0","batch_idx % SAVE_STEPS == 0")
            namespace =dict (trainer .__dict__ )
            namespace ["SAVE_STEPS"]=save_steps 
            exec (patched_src ,namespace )
            patched_epoch =namespace ["train_epoch"]
        else :
            patched_epoch =original 
    except Exception :
        patched_epoch =original 

    def _patched (*args ,start_batch =0 ,lr_scheduler =None ,**kwargs ):
        return train_epoch_with_resume (patched_epoch ,*args ,start_batch =start_batch ,lr_scheduler =lr_scheduler ,**kwargs )

    trainer .train_epoch =_patched 
    trainer ._resume_patch_applied =True 

