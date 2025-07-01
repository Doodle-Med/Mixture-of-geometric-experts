#!/usr/bin/env python3


"""
MGM Flagship Production Training Script
======================================

Production-scale training with:
- 32K context window (Qwen 1.5 scale)
- 64 geometric experts with 8-way routing
- 20 tokens/parameter ratio (40B tokens target)
- Full multimodal pipeline (13 datasets)
- ComplexFilteredGradScaler for stable training
- Comprehensive monitoring and checkpointing

Architecture: ~2B parameters
Context: 32,768 tokens
Experts: 64 geometric manifolds
Target: 40B training tokens
"""

import os 
import sys 
import json 
import time 
import logging 
import subprocess 
from pathlib import Path 
import argparse 
import torch 
import psutil 
import GPUtil 


project_root =Path (__file__ ).resolve ().parent 
sys .path .insert (0 ,str (project_root ))

import train_geometric_model_v2 as trainer 
from resume_helper import patch_trainer_for_resume 


logging .basicConfig (
level =logging .INFO ,
format ='%(asctime)s - %(levelname)s - %(message)s',
handlers =[
logging .FileHandler ('flagship_production.log'),
logging .StreamHandler (sys .stdout )
]
)
logger =logging .getLogger (__name__ )

class ProductionEnvironmentSetup :
    """Handles production environment setup and optimization."""

    def __init__ (self ):
        self .gpu_info =self ._get_gpu_info ()
        self .cpu_info =self ._get_cpu_info ()
        self .memory_info =self ._get_memory_info ()

    def _get_gpu_info (self ):
        """Get GPU information."""
        try :
            gpus =GPUtil .getGPUs ()
            return {
            'count':len (gpus ),
            'names':[gpu .name for gpu in gpus ],
            'memory_total':[gpu .memoryTotal for gpu in gpus ],
            'memory_free':[gpu .memoryFree for gpu in gpus ]
            }
        except :
            return {'count':0 ,'names':[],'memory_total':[],'memory_free':[]}

    def _get_cpu_info (self ):
        """Get CPU information."""
        return {
        'count':psutil .cpu_count (),
        'freq':psutil .cpu_freq ().current if psutil .cpu_freq ()else None 
        }

    def _get_memory_info (self ):
        """Get memory information."""
        mem =psutil .virtual_memory ()
        return {
        'total':mem .total //(1024 **3 ),
        'available':mem .available //(1024 **3 ),
        'percent':mem .percent 
        }

    def setup_environment (self ):
        """Setup production environment."""
        logger .info ("🚀 Setting up flagship production environment...")


        logger .info (f"💻 System Info:")
        logger .info (f"   CPUs: {self .cpu_info ['count']}")
        logger .info (f"   Memory: {self .memory_info ['total']}GB total, {self .memory_info ['available']}GB available")
        logger .info (f"   GPUs: {self .gpu_info ['count']}")

        for i ,(name ,total ,free )in enumerate (zip (
        self .gpu_info ['names'],
        self .gpu_info ['memory_total'],
        self .gpu_info ['memory_free']
        )):
            logger .info (f"   GPU {i }: {name } ({total }MB total, {free }MB free)")


        os .environ ['CUDA_LAUNCH_BLOCKING']='0'
        os .environ ['TORCH_CUDNN_V8_API_ENABLED']='1'
        os .environ ['TOKENIZERS_PARALLELISM']='true'


        if torch .cuda .is_available ():
            torch .cuda .empty_cache ()

            os .environ ['PYTORCH_CUDA_ALLOC_CONF']='max_split_size_mb:512'


        torch .multiprocessing .set_start_method ('spawn',force =True )

        logger .info ("✅ Environment setup complete")

    def install_dependencies (self ):
        """Install required dependencies."""
        logger .info ("📦 Installing/updating dependencies...")

        dependencies =[
        'torch>=2.0.0',
        'transformers>=4.30.0',
        'datasets>=2.12.0',
        'accelerate>=0.20.0',
        'wandb>=0.15.0',
        'geoopt>=0.5.0',
        'flash-attn>=2.0.0',
        'deepspeed>=0.9.0'
        ]

        for dep in dependencies :
            try :
                subprocess .run ([sys .executable ,'-m','pip','install',dep ],
                check =True ,capture_output =True )
                logger .info (f"✅ {dep }")
            except subprocess .CalledProcessError as e :
                logger .warning (f"⚠️ Failed to install {dep }: {e }")

        logger .info ("✅ Dependencies installation complete")

class FlagshipModelCalculator :
    """Calculate model parameters and training requirements."""

    @staticmethod 
    def calculate_parameters (config ):
        """Calculate approximate model parameters."""
        model_config =config ['model']


        vocab_size =model_config ['vocab_size']
        input_dim =model_config ['input_dim']
        embedding_params =vocab_size *input_dim 


        num_experts =model_config ['num_experts']
        hidden_dim =model_config ['hidden_dim']
        output_dim =model_config ['output_dim']


        expert_params =num_experts *(
        input_dim *hidden_dim +
        hidden_dim *output_dim 
        )


        gate_params =input_dim *num_experts 


        memory_slots =model_config ['memory_slots']
        memory_width =model_config ['memory_width']
        memory_params =memory_slots *memory_width *3 


        thought_params =4 *(input_dim *hidden_dim )


        final_head_params =output_dim *vocab_size 

        total_params =(
        embedding_params +expert_params +gate_params +
        memory_params +thought_params +final_head_params 
        )

        return {
        'total':total_params ,
        'embedding':embedding_params ,
        'experts':expert_params ,
        'gating':gate_params ,
        'memory':memory_params ,
        'thought_generator':thought_params ,
        'final_head':final_head_params 
        }

    @staticmethod 
    def calculate_training_requirements (config ,param_count ):
        """Calculate training token requirements."""
        tokens_per_param =config ['training']['tokens_per_parameter']
        target_tokens =param_count *tokens_per_param 

        seq_len =config ['data']['seq_len']
        batch_size =config ['training']['batch_size']
        grad_accum =config ['training']['gradient_accumulation_steps']

        effective_batch_size =batch_size *grad_accum 
        tokens_per_step =effective_batch_size *seq_len 
        required_steps =target_tokens //tokens_per_step 

        return {
        'target_tokens':target_tokens ,
        'tokens_per_step':tokens_per_step ,
        'required_steps':required_steps ,
        'effective_batch_size':effective_batch_size 
        }

class FlagshipTrainingOrchestrator :
    """Orchestrates flagship production training."""

    def __init__ (self ,config_path :Path ):
        self .config_path =config_path 
        self .config =self ._load_config ()
        self .env_setup =ProductionEnvironmentSetup ()
        self .start_time =time .time ()

    def _load_config (self ):
        """Load configuration file."""
        with open (self .config_path ,'r')as f :
            return json .load (f )

    def _log_model_info (self ):
        """Log model architecture and training information."""
        calc =FlagshipModelCalculator ()
        param_info =calc .calculate_parameters (self .config )
        training_info =calc .calculate_training_requirements (self .config ,param_info ['total'])

        logger .info ("🏗️ Flagship Model Architecture:")
        logger .info (f"   📊 Total Parameters: {param_info ['total']:,}")
        logger .info (f"   🧠 Experts: {self .config ['model']['num_experts']}")
        logger .info (f"   🎯 Routing (k): {self .config ['model']['k']}")
        logger .info (f"   📏 Context Length: {self .config ['data']['seq_len']:,}")
        logger .info (f"   🔄 Recursion Steps: {self .config ['model']['recursion_steps']}")
        logger .info (f"   💾 Memory Slots: {self .config ['model']['memory_slots']}")

        logger .info ("🎯 Training Requirements:")
        logger .info (f"   📈 Target Tokens: {training_info ['target_tokens']:,}")
        logger .info (f"   📦 Effective Batch Size: {training_info ['effective_batch_size']}")
        logger .info (f"   🔢 Required Steps: {training_info ['required_steps']:,}")
        logger .info (f"   ⏱️ Tokens/Step: {training_info ['tokens_per_step']:,}")

        return param_info ,training_info 

    def setup_monitoring (self ):
        """Setup monitoring and logging."""
        try :

            import os 
            if os .environ .get ('WANDB_DISABLED','').lower ()=='true':
                logger .info ("✅ WandB monitoring disabled via environment variable")
                return 


            wandb_project =self .config .get ('monitoring',{}).get ('wandb_project')
            if not wandb_project :
                logger .info ("✅ WandB monitoring disabled in configuration")
                return 

            import wandb 
            if not hasattr (wandb ,'init'):
                logger .warning ("⚠️ wandb.init not available, skipping monitoring setup")
                return 

            wandb .init (
            project =wandb_project ,
            config =self .config ,
            name =f"flagship-production-{int (time .time ())}",
            tags =["flagship","production","multimodal","32k-context"]
            )
            logger .info ("✅ Weights & Biases monitoring initialized")
        except (ImportError ,AttributeError )as e :
            logger .warning (f"⚠️ wandb not available ({e }), skipping monitoring setup")

    def run_training (self ):
        """Execute the flagship training."""
        logger .info ("🚀 Starting flagship production training...")


        patch_trainer_for_resume (trainer )


        args =[
        "train_geometric_model_v2.py",
        "--config",
        str (self .config_path ),
        "--end_to_end",
        ]

        if self .config .get ("use_streaming",True ):
            args .append ("--streaming")
            logger .info ("✅ Using streaming dataloaders")
        else :
            logger .info ("✅ Using local NPZ/NPY dataloaders (streaming disabled)")

        sys .argv =args 

        try :

            trainer .main ()

            elapsed =time .time ()-self .start_time 
            logger .info (f"🎉 Flagship training completed successfully!")
            logger .info (f"⏱️ Total training time: {elapsed /3600 :.2f} hours")

        except Exception as e :
            logger .error (f"❌ Training failed: {e }")
            import traceback 
            traceback .print_exc ()
            raise 

    def create_deployment_package (self ):
        """Create deployment package with model and configs."""
        logger .info ("📦 Creating deployment package...")

        deployment_dir =Path ("flagship_deployment")
        deployment_dir .mkdir (exist_ok =True )


        import shutil 
        shutil .copy2 (self .config_path ,deployment_dir /"config.json")


        deployment_info ={
        "model_type":"MGM Flagship",
        "parameters":"~2B",
        "context_length":self .config ['data']['seq_len'],
        "experts":self .config ['model']['num_experts'],
        "created_at":time .strftime ("%Y-%m-%d %H:%M:%S"),
        "training_time_hours":(time .time ()-self .start_time )/3600 
        }

        with open (deployment_dir /"deployment_info.json","w")as f :
            json .dump (deployment_info ,f ,indent =2 )

        logger .info (f"✅ Deployment package created: {deployment_dir }")

def main ():
    """Main entry point for flagship production training."""
    parser =argparse .ArgumentParser (
    description ="MGM Flagship Production Training",
    formatter_class =argparse .RawDescriptionHelpFormatter ,
    epilog ="""
🚀 Flagship Production Training Features:
- 32K context window (Qwen 1.5 scale)
- 64 geometric experts with 8-way routing  
- 20 tokens/parameter ratio (40B tokens)
- 13 multimodal datasets
- ComplexFilteredGradScaler for stability
- Production monitoring and checkpointing

Example usage:
  python run_flagship_production.py --config production_flagship_config.json
        """
    )

    parser .add_argument (
    "--config",
    type =str ,
    default ="production_flagship_config.json",
    help ="Configuration file path"
    )
    parser .add_argument (
    "--setup-only",
    action ="store_true",
    help ="Only setup environment without training"
    )
    parser .add_argument (
    "--skip-deps",
    action ="store_true",
    help ="Skip dependency installation"
    )

    args =parser .parse_args ()


    config_path =Path (args .config )
    if not config_path .exists ():
        logger .error (f"❌ Configuration file not found: {config_path }")
        sys .exit (1 )

    orchestrator =FlagshipTrainingOrchestrator (config_path )

    try :

        orchestrator .env_setup .setup_environment ()

        if not args .skip_deps :
            orchestrator .env_setup .install_dependencies ()


        orchestrator ._log_model_info ()

        if args .setup_only :
            logger .info ("✅ Environment setup complete. Use --setup-only=false to start training.")
            return 


        orchestrator .setup_monitoring ()


        orchestrator .run_training ()


        orchestrator .create_deployment_package ()

        logger .info ("🎉 Flagship production training pipeline completed successfully!")

    except KeyboardInterrupt :
        logger .info ("⚠️ Training interrupted by user")
        sys .exit (1 )
    except Exception as e :
        logger .error (f"❌ Pipeline failed: {e }")
        sys .exit (1 )

if __name__ =="__main__":
    main ()
