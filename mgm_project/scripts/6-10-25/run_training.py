#!/usr/bin/env python3


"""End-to-end training launcher for MGM

This script replicates the behavior of the original notebook
``actual_working_version.py`` but in a plain Python form so it can
be executed outside of Jupyter. It generates an optimised configuration
and wrapper script, then launches ``train_geometric_model_v2.py`` with
checkpoint resume logic.
"""
import json 
import os 
import subprocess 
import textwrap 
from pathlib import Path 
import argparse 

def build_config (config_path :Path )->None :
    """Write the optimised training configuration."""
    optimized_config ={
    "vocab_size":50257 ,
    "model":{
    "vocab_size":50257 ,
    "input_dim":512 ,
    "hidden_dim":1024 ,
    "output_dim":512 ,
    "final_output_dim":50257 ,
    "pad_token_id":50256 ,
    "num_experts":16 ,
    "k":4 ,
    "recursion_steps":2 ,
    "memory_slots":64 ,
    "memory_width":1536 ,
    "manifolds":[
    "euclidean","hyperbolic","spherical","poincare",
    "simplex","complex","lorentzian","product"
    ]*2 ,
    },
    "data":{"seq_len":1024 ,"val_split":0.05 },
    "streaming":{
    "seq_len":1024 ,
    "pad_to_multiple_of":8 ,
    "tokenizer_name":"gpt2",
    "modalities":{

    "text":{
    "hf_dataset_name":"wikitext",
    "config_name":"wikitext-103-raw-v1",
    "text_column":"text",
    "sampling_ratio":0.20 ,
    },
    "web":{
    "hf_dataset_name":"allenai/c4",
    "config_name":"en",
    "text_column":"text",
    "sampling_ratio":0.15 ,
    },
    "books":{
    "hf_dataset_name":"bookcorpus",
    "config_name":None ,
    "text_column":"text",
    "sampling_ratio":0.05 ,
    "trust_remote_code":True ,
    },


    "code":{
    "hf_dataset_name":"bigcode/the-stack-dedup",
    "config_name":"default",
    "text_column":"content",
    "sampling_ratio":0.10 ,
    },
    "code_python":{
    "hf_dataset_name":"codeparrot/github-code",
    "config_name":"python",
    "text_column":"code",
    "sampling_ratio":0.05 ,
    },


    "scientific":{
    "hf_dataset_name":"scientific_papers",
    "config_name":"pubmed",
    "text_column":"abstract",
    "sampling_ratio":0.08 ,
    "trust_remote_code":True ,
    },
    "arxiv":{
    "hf_dataset_name":"scientific_papers",
    "config_name":"arxiv",
    "text_column":"abstract",
    "sampling_ratio":0.05 ,
    "trust_remote_code":True ,
    },
    "patents":{
    "hf_dataset_name":"bigpatent",
    "config_name":"all",
    "text_column":"abstract",
    "sampling_ratio":0.02 ,
    },


    "cot":{
    "hf_dataset_name":"allenai/math_qa",
    "config_name":None ,
    "text_column":"Problem",
    "sampling_ratio":0.05 ,
    },
    "reasoning":{
    "hf_dataset_name":"squad",
    "config_name":None ,
    "text_column":"question",
    "sampling_ratio":0.05 ,
    },
    "logic_reasoning":{
    "hf_dataset_name":"glue",
    "config_name":"rte",
    "text_column":"sentence1",
    "sampling_ratio":0.03 ,
    },
    "commonsense":{
    "hf_dataset_name":"commonsense_qa",
    "config_name":None ,
    "text_column":"question",
    "sampling_ratio":0.02 ,
    },


    "conversational":{
    "hf_dataset_name":"daily_dialog",
    "config_name":None ,
    "text_column":"dialog",
    "sampling_ratio":0.05 ,
    "trust_remote_code":True ,
    },
    "persona_chat":{
    "hf_dataset_name":"conv_ai_2",
    "config_name":None ,
    "text_column":"text",
    "sampling_ratio":0.05 ,
    },


    "audio":{
    "hf_dataset_name":"openslr/librispeech_asr",
    "config_name":"clean",
    "split":"train.100",
    "text_column":"text",
    "audio_column":"audio",
    "sampling_ratio":0.03 ,
    "trust_remote_code":True ,
    },
    "image":{
    "hf_dataset_name":"cifar10",
    "config_name":None ,
    "text_column":"label",
    "image_column":"img",
    "sampling_ratio":0.02 ,
    },
    },
    },
    "training":{
    "batch_size":1 ,
    "gradient_accumulation_steps":32 ,
    "learning_rate":1e-4 ,
    "max_steps":50000 ,
    "warmup_steps":500 ,
    "weight_decay":0.01 ,
    "max_grad_norm":1.0 ,
    "dataloader_num_workers":2 ,
    "save_steps":1000 ,
    "eval_steps":500 ,
    "logging_steps":50 ,
    "fp16":True ,
    "use_amp":True ,
    "dataloader_pin_memory":True ,
    "gradient_checkpointing":True ,
    "optim":"adamw_torch",
    "lr_scheduler_type":"cosine",
    "save_total_limit":3 ,
    "use_flash_attention":True ,
    "balance_loss_weight":0.05 ,
    "curvature_reg_weight":0.005 ,
    "verifier_loss_weight":0.02 ,
    "memory_auxiliary_loss_weight":0.1 ,
    "evaluation_strategy":"steps",
    "save_strategy":"steps",
    "seed":42 ,
    },
    "training_stages":{
    "curvature_calibration":{
    "epochs":1 ,
    "label_smoothing":0.0 ,
    "optimizer":{
    "learning_rate":3e-3 ,
    "geo_lr_factor":1.0 ,
    "weight_decay":0.0 ,
    "reward_lr":0.0 ,
    },
    },
    "reasoning_warmup":{
    "epochs":2 ,
    "label_smoothing":0.05 ,
    "optimizer":{
    "learning_rate":1e-3 ,
    "reward_lr":5e-4 ,
    "weight_decay":0.005 ,
    "geo_lr_factor":1.0 ,
    },
    },
    "branch_pretrain":{
    "epochs":3 ,
    "label_smoothing":0.1 ,
    "optimizer":{
    "learning_rate":8e-4 ,
    "reward_lr":4e-4 ,
    "weight_decay":0.01 ,
    "geo_lr_factor":1.5 ,
    },
    },
    "gate_train":{
    "epochs":2 ,
    "label_smoothing":0.1 ,
    "optimizer":{
    "learning_rate":2e-4 ,
    "reward_lr":1e-4 ,
    "weight_decay":0.01 ,
    "geo_lr_factor":1.0 ,
    },
    },
    "joint_finetune":{
    "epochs":3 ,
    "label_smoothing":0.15 ,
    "scheduler_t0":2 ,
    "optimizer":{
    "learning_rate":1e-4 ,
    "reward_lr":5e-5 ,
    "weight_decay":0.015 ,
    "geo_lr_factor":1.2 ,
    },
    },
    },
    }
    with open (config_path ,"w")as f :
        json .dump (optimized_config ,f ,indent =2 )


def build_wrapper (wrapper_path :Path ,config_path :Path ,checkpoint_path :Path ,project_dir :Path )->None :
    """Create a wrapper script that patches the trainer and launches it."""
    wrapper_script =textwrap .dedent (
    f"""
        import sys, os, gc
        import torch
        from torch.utils.checkpoint import checkpoint

        project_dir = r"{project_dir }"
        if project_dir not in sys.path:
            sys.path.insert(0, project_dir)

        gc.collect()
        torch.cuda.empty_cache()

        from train_geometric_model_v2 import GeometricExpert, ThoughtGenerator, main as trainer_main

        for cls in (GeometricExpert, ThoughtGenerator):
            orig_forward = cls.forward
            cls.forward = lambda self, x, _orig=orig_forward: checkpoint(_orig, self, x, use_reentrant=True)

        sys.argv = ["train_geometric_model_v2.py", "--config", str(config_path), "--streaming", "--end_to_end"]
        if os.path.exists(str(checkpoint_path)):
            sys.argv += ["--checkpoint", str(checkpoint_path)]

        trainer_main()
        """
    )
    with open (wrapper_path ,"w")as f :
        f .write (wrapper_script )


def launch (wrapper_path :Path ,log_file :Path ,dry_run :bool =False )->None :
    """Execute the wrapper script in a subprocess."""
    if dry_run :
        print ("[DRY RUN] Wrapper script prepared at",wrapper_path )
        return 
    with open (log_file ,"w")as logf :
        proc =subprocess .Popen (["python",str (wrapper_path )],stdout =logf ,stderr =subprocess .STDOUT )
    proc .wait ()


def main ()->None :
    parser =argparse .ArgumentParser (description ="Launch MGM training")
    parser .add_argument ("--dry-run",action ="store_true",help ="Only create scripts without execution")
    args =parser .parse_args ()

    project_dir =Path (__file__ ).resolve ().parent 
    config_path =project_dir /"optim_run_cfg.json"
    wrapper_path =project_dir /"run_optimized.py"
    log_file =project_dir /"train_run_resumed.log"
    checkpoint =project_dir /"checkpoint_epoch_0_batch_165000.pth.tar"

    build_config (config_path )
    build_wrapper (wrapper_path ,config_path ,checkpoint ,project_dir )
    print ("✅ Wrapper script and config created.")

    launch (wrapper_path ,log_file ,dry_run =args .dry_run )


if __name__ =="__main__":
    main ()

