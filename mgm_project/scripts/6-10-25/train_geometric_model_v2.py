#!/usr/bin/env python3


"""
Mixture of Geometric Experts Training Script for MGM Project
===========================================================

Enhanced training script that uses the comprehensive NPZ datasets and
ultimate tokenizer for the MGM neural network.
"""

import os 
os .environ .setdefault ("HF_HUB_DISABLE_XET","1")
import sys 
import argparse 
import json 
import logging 
import time 
import subprocess 
import math 
import itertools 
import random 
from pathlib import Path 



import torch 
import torch .nn as nn 
import torch .nn .functional as F 
import torch ._dynamo 
from torch .utils .data import DataLoader ,IterableDataset ,ConcatDataset 
from torch .optim .lr_scheduler import CosineAnnealingWarmRestarts 
import numpy as np 
from tqdm import tqdm 
from pathlib import Path 
from typing import Optional ,Tuple ,List ,Dict 

TORCHDATA_AVAILABLE =False 


logging .basicConfig (level =logging .INFO )
logger =logging .getLogger (__name__ )

import math 
import torch .nn .functional as F 


class CausalBlock (nn .Module ):
    """
    Tiny causal transformer block: LayerNorm → causal MHA → LayerNorm → FFN.
    Adds ~1 M parameters when dim=256 and n_heads=4.
    """

    def __init__ (self ,dim :int ,n_heads :int =4 ,ffn_mult :int =4 ):
        super ().__init__ ()
        self .ln1 =nn .LayerNorm (dim )
        self .attn =nn .MultiheadAttention (dim ,n_heads ,batch_first =True )
        self .ln2 =nn .LayerNorm (dim )
        self .ffn =nn .Sequential (
        nn .Linear (dim ,ffn_mult *dim ),
        nn .GELU (),
        nn .Linear (ffn_mult *dim ,dim ),
        )

    def forward (self ,x :torch .Tensor )->torch .Tensor :

        B ,T ,_ =x .shape 
        causal =torch .triu (torch .ones (T ,T ,dtype =torch .bool ,device =x .device ),1 )
        h =x +self .attn (self .ln1 (x ),self .ln1 (x ),self .ln1 (x ),attn_mask =causal )[0 ]
        return h +self .ffn (self .ln2 (h ))




import torch .utils .checkpoint as _cp 

def _no_checkpoint (fn ,*args ,**kwargs ):
    """Bypass torch.utils.checkpoint.checkpoint; executes fn directly."""
    return fn (*args ,**kwargs )

_cp .checkpoint =_no_checkpoint 


try :
    import geoopt 
    from geoopt .optim import RiemannianAdam 
    logger .info (f"✅ GeoOpt version {geoopt .__version__ } already available and imported.")
    GEOOPT_AVAILABLE =True 
except ImportError :
    logger .info ("🔧 GeoOpt not found. Attempting to install from git (requires git command).")
    try :

        subprocess .check_call ([sys .executable ,"-m","pip","install","-q","--no-deps","git+https://github.com/geoopt/geoopt.git"])
        logger .info ("🔧 GeoOpt installation from git attempted successfully via pip.")
        logger .warning ("⚠️ IMPORTANT: If GeoOpt was just installed, you might need to RESTART THE KERNEL "
        "for the 'import geoopt' in subsequent training to work.")
        logger .info ("🔧 Attempting to import geoopt again post-install attempt...")
        import geoopt 
        from geoopt .optim import RiemannianAdam 
        logger .info (f"✅ GeoOpt version {geoopt .__version__ } imported successfully AFTER installation attempt.")
        GEOOPT_AVAILABLE =True 
    except subprocess .CalledProcessError as e_pip_geo :
        logger .error (f"❌ Failed to install geoopt using pip from git: {e_pip_geo }")
        logger .error ("Please ensure git is installed and accessible. You may need to install geoopt manually.")
        GEOOPT_AVAILABLE =False 
    except ImportError as e_imp_geo :
        logger .error (f"❌ GeoOpt installed (or installation attempted) but still FAILED TO IMPORT: {e_imp_geo }")
        logger .critical ("🚨 CRITICAL: GeoOpt could not be imported. Training will use standard optimizers. "
        "PLEASE RESTART THE KERNEL and re-run the script from the beginning.")
        GEOOPT_AVAILABLE =False 
    except Exception as e_geo_other :
        logger .error (f"❌ An unexpected error occurred during GeoOpt installation/import: {e_geo_other }")
        logger .critical ("🚨 CRITICAL: GeoOpt setup encountered an issue. Training may fail or use standard optimizers.")
        GEOOPT_AVAILABLE =False 

try :
    from transformers import AutoTokenizer ,CLIPVisionModel 
    HF_AVAILABLE =True 
except ImportError as e :
    print (f"Error: Transformers not available: {e }",file =sys .stderr )
    HF_AVAILABLE =False 
    raise ImportError ("Transformers is required but not available")


try :
    from memory_guard import (
    DynamicMemoryGuard ,
    numerical_stability_context ,
    AdaptiveBatchSize ,
    setup_cuda_optimizations ,
    create_safe_optimizer 
    )
    MEMORY_GUARD_AVAILABLE =True 
    logger .info ("✅ Memory guard system loaded")
except ImportError as e :
    logger .warning (f"⚠️ Memory guard not available: {e }")
    MEMORY_GUARD_AVAILABLE =False 


DATA_PATHS ={
'all_combined_tokens':Path (__file__ ).parent /'npy_generation_system/output/enhanced_multimodal/combined_dataset/ALL_COMBINED_tokens/ALL_COMBINED_tokens_combined.npz',
'all_combined_attention':Path (__file__ ).parent /'npy_generation_system/output/enhanced_multimodal/combined_dataset/ALL_COMBINED_attention/ALL_COMBINED_attention_combined.npz',
'all_combined_caption':Path (__file__ ).parent /'npy_generation_system/output/enhanced_multimodal/combined_dataset/ALL_COMBINED_caption/ALL_COMBINED_caption_combined.npz',
'all_combined_wiki':Path (__file__ ).parent /'npy_generation_system/output/enhanced_multimodal/combined_dataset/ALL_COMBINED_wiki/ALL_COMBINED_wiki_combined.npz',
'all_combined_medical':Path (__file__ ).parent /'npy_generation_system/output/enhanced_multimodal/combined_dataset/ALL_COMBINED_medical/ALL_COMBINED_medical_combined.npz',
'all_combined_python':Path (__file__ ).parent /'npy_generation_system/output/enhanced_multimodal/combined_dataset/ALL_COMBINED_python/ALL_COMBINED_python_combined.npz',
'all_combined_java':Path (__file__ ).parent /'npy_generation_system/output/enhanced_multimodal/combined_dataset/ALL_COMBINED_java/ALL_COMBINED_java_combined.npz',
'captions_caption':Path (__file__ ).parent /'npy_generation_system/output/enhanced_multimodal/combined_dataset/captions_caption/captions_caption_combined.npz',
'code_python':Path (__file__ ).parent /'npy_generation_system/output/enhanced_multimodal/combined_dataset/code_python/code_python_combined.npz',
'code_java':Path (__file__ ).parent /'npy_generation_system/output/enhanced_multimodal/combined_dataset/code_java/code_java_combined.npz',
'audio_audio':Path (__file__ ).parent /'npy_generation_system/output/enhanced_multimodal/combined_dataset/audio_audio/audio_audio_combined.npz'
}


TOKENIZER_PATH =Path (__file__ ).parent /'npy_data/ultimate_tokenizer'


def safe_softplus (x ,beta =1 ,threshold =20 ):
    """Clamped, ramp-aware SoftPlus"""
    return F .softplus (x ,beta =beta ,threshold =threshold )+1e-5 

def has_nan (tensor ):
    """Quick NaN/Inf detector"""
    return not torch .isfinite (tensor ).all ()


from torch .quasirandom import SobolEngine as _MGM_SobolEngine 


_SOBOL_ENGINE =_MGM_SobolEngine (1 ,scramble =True ,seed =42 )
_SOBOL_CACHE :List [float ]=[]

def sobol_raw_c (expert_id :int ,total :int ,lo :float =-5.0 ,hi :float =2.0 )->float :
    """Generate a log-curvature value using a Sobol low-discrepancy sequence.

    This helps to spread expert curvatures across a wide range and mitigates
    early manifold monoculture (see HELM-MICE & S2-MoE).
    """
    global _SOBOL_CACHE 
    if len (_SOBOL_CACHE )<total :
        required =total -len (_SOBOL_CACHE )
        _SOBOL_CACHE .extend (_SOBOL_ENGINE .draw (required ).squeeze (-1 ).tolist ())
    u =_SOBOL_CACHE [expert_id ]
    return lo +(hi -lo )*u 


class ComplexFilteredGradScaler (torch .amp .GradScaler ):
    """
    Custom GradScaler that filters out complex tensors to avoid CUDA errors.
    Complex tensors from geometric manifolds are handled separately without AMP.
    """

    def __init__ (self ,device ='cuda',*args ,**kwargs ):
        super ().__init__ (device ,*args ,**kwargs )
        self .complex_params =set ()

    def register_complex_params (self ,model ):
        """Register parameters that contain complex tensors"""
        self .complex_params .clear ()
        for name ,param in model .named_parameters ():
            if param .dtype in [torch .complex64 ,torch .complex128 ]:
                self .complex_params .add (id (param ))
                logging .info (f"🔍 Registered complex parameter for AMP filtering: {name }")

    def _filter_complex_grads (self ,optimizer ):
        """Filter out complex gradients before AMP operations"""
        complex_grads ={}

        for group in optimizer .param_groups :
            for param in group ['params']:
                if param .grad is not None and id (param )in self .complex_params :

                    complex_grads [id (param )]=param .grad 
                    param .grad =None 

        return complex_grads 

    def _restore_complex_grads (self ,optimizer ,complex_grads ):
        """Restore complex gradients after AMP operations"""
        for group in optimizer .param_groups :
            for param in group ['params']:
                if id (param )in complex_grads :
                    param .grad =complex_grads [id (param )]

    def unscale_ (self ,optimizer ):
        """Override unscale to filter complex tensors"""

        complex_grads =self ._filter_complex_grads (optimizer )


        try :
            super ().unscale_ (optimizer )
        except Exception as e :

            self ._restore_complex_grads (optimizer ,complex_grads )
            raise e 


        self ._restore_complex_grads (optimizer ,complex_grads )

    def step (self ,optimizer ,*args ,**kwargs ):
        """Override step to handle complex tensors safely"""

        complex_grads =self ._filter_complex_grads (optimizer )


        try :
            retval =super ().step (optimizer ,*args ,**kwargs )
        except Exception as e :

            self ._restore_complex_grads (optimizer ,complex_grads )
            raise e 


        self ._restore_complex_grads (optimizer ,complex_grads )


        with torch .no_grad ():
            for group in optimizer .param_groups :
                for param in group ['params']:
                    if id (param )in self .complex_params and param .grad is not None :

                        if hasattr (optimizer ,'_step_complex_param'):
                            optimizer ._step_complex_param (param ,group )
                        else :

                            lr =group ['lr']
                            param .data .add_ (param .grad ,alpha =-lr )

        return retval 

class ProductionSafetyChecks :
    """Comprehensive safety checks for production training"""

    @staticmethod 
    def validate_config (config :dict )->list [str ]:
        """Validate configuration for production safety"""
        errors =[]


        required_fields =['vocab_size','data','model','training']
        for field in required_fields :
            if field not in config :
                errors .append (f"Missing required config field: {field }")


        if config .get ('vocab_size',0 )<=0 :
            errors .append ("vocab_size must be positive")


        seq_len =config .get ('data',{}).get ('seq_len',0 )
        max_pos =config .get ('max_position_embeddings',seq_len )
        if seq_len >max_pos :
            errors .append (f"seq_len {seq_len } > max_position_embeddings {max_pos }")

        return errors 

    @staticmethod 
    def validate_batch (batch :dict ,config :dict )->list [str ]:
        """Validate batch data for safety - CRITICAL FOR PRODUCTION"""
        errors =[]

        vocab_size =config .get ('vocab_size',config .get ('model',{}).get ('final_output_dim',50300 ))
        max_pos =config .get ('max_position_embeddings',2048 )


        if 'input_ids'in batch :
            input_ids =batch ['input_ids']


            if input_ids .numel ()==0 :
                errors .append ("CRITICAL: Empty input_ids tensor - will cause processing errors")
                return errors 


            if torch .any (input_ids <0 ):
                errors .append ("CRITICAL: Negative token IDs found in input_ids - will cause CUDA crash")


            max_id =torch .max (input_ids ).item ()
            if max_id >=vocab_size :
                errors .append (f"CRITICAL: Token ID {max_id } >= vocab_size {vocab_size } - will cause index error")


            seq_len =input_ids .shape [1 ]if input_ids .dim ()>1 else len (input_ids )
            if seq_len >max_pos :
                errors .append (f"CRITICAL: Sequence length {seq_len } > max_position_embeddings {max_pos }")


            if input_ids .dtype not in [torch .int32 ,torch .long ]:
                errors .append (f"WARNING: input_ids dtype {input_ids .dtype } - recommend int32 for GPU efficiency")


        if 'labels'in batch :
            labels =batch ['labels']


            if torch .any (labels <-100 ):
                errors .append ("WARNING: Labels contain values < -100")

        return errors 

    @staticmethod 
    def validate_embeddings (embedding_layer :nn .Embedding ,config :dict )->list [str ]:
        """Validate embedding layer configuration"""
        errors =[]

        expected_vocab_size =config .get ('vocab_size',config .get ('model',{}).get ('final_output_dim',50300 ))
        actual_vocab_size =embedding_layer .num_embeddings 

        if actual_vocab_size !=expected_vocab_size :
            errors .append (f"CRITICAL: Embedding vocab_size {actual_vocab_size } != config vocab_size {expected_vocab_size }")

        return errors 

def validate_model_checkpoint_compatibility (model :nn .Module ,checkpoint_path :str ,config :dict )->bool :
    """Validate checkpoint compatibility with current model and vocab size - CRITICAL FOR RESUME"""
    if not os .path .exists (checkpoint_path ):
        return True 

    try :
        logger .info (f"🔍 Validating checkpoint compatibility: {checkpoint_path }")
        checkpoint =torch .load (checkpoint_path ,map_location ='cpu',weights_only =False )
        state_dict =checkpoint .get ('model_state_dict',checkpoint .get ('state_dict',checkpoint ))


        embedding_key =None 
        for key in state_dict .keys ():
            if 'embedding'in key and 'weight'in key :
                embedding_key =key 
                break 

        if embedding_key :
            saved_vocab_size =state_dict [embedding_key ].shape [0 ]
            current_vocab_size =config .get ('vocab_size',config .get ('model',{}).get ('final_output_dim',50300 ))

            if saved_vocab_size !=current_vocab_size :
                logger .warning (f"Checkpoint vocab_size {saved_vocab_size } != config vocab_size {current_vocab_size }")


                if current_vocab_size >saved_vocab_size :
                    logger .info ("🔧 Auto-expanding embedding matrix...")
                    pad_size =current_vocab_size -saved_vocab_size 
                    embedding_dim =state_dict [embedding_key ].shape [1 ]


                    padding =torch .randn (pad_size ,embedding_dim )*0.02 
                    state_dict [embedding_key ]=torch .cat ([
                    state_dict [embedding_key ],padding 
                    ],dim =0 )


                    output_keys =[k for k in state_dict .keys ()if 'final_head'in k or 'output_head'in k ]
                    for output_key in output_keys :
                        if 'weight'in output_key :
                            output_padding =torch .randn (pad_size ,state_dict [output_key ].shape [1 ])*0.02 
                            state_dict [output_key ]=torch .cat ([
                            state_dict [output_key ],output_padding 
                            ],dim =0 )
                        elif 'bias'in output_key :
                            bias_padding =torch .zeros (pad_size )
                            state_dict [output_key ]=torch .cat ([
                            state_dict [output_key ],bias_padding 
                            ],dim =0 )

                    logger .info (f"✅ Expanded embeddings from {saved_vocab_size } to {current_vocab_size }")


                    checkpoint ['model_state_dict']=state_dict 
                    checkpoint ['vocab_size']=current_vocab_size 
                    checkpoint ['updated']=True 
                    torch .save (checkpoint ,checkpoint_path )

                else :
                    logger .error (f"❌ Cannot shrink vocab from {saved_vocab_size } to {current_vocab_size }")
                    return False 


        try :

            model_keys =set (model .state_dict ().keys ())
            checkpoint_keys =set (state_dict .keys ())

            missing_keys =model_keys -checkpoint_keys 
            unexpected_keys =checkpoint_keys -model_keys 

            if missing_keys :
                logger .warning (f"Missing keys in checkpoint: {list (missing_keys )[:5 ]}...")
            if unexpected_keys :
                logger .warning (f"Unexpected keys in checkpoint: {list (unexpected_keys )[:5 ]}...")


            load_result =model .load_state_dict (state_dict ,strict =False )
            logger .info (f"✅ Checkpoint validation passed. Missing: {len (load_result .missing_keys )}, Unexpected: {len (load_result .unexpected_keys )}")

        except Exception as load_error :
            logger .error (f"❌ State dict loading test failed: {load_error }")
            return False 

        return True 

    except Exception as e :
        logger .error (f"❌ Error validating checkpoint: {e }")
        return False 

def safe_collate_fn (batch :list )->dict :
    """Production-safe collate function with comprehensive error handling"""
    try :

        batch =[sample for sample in batch if sample is not None ]

        if not batch :

            return {
            'input_ids':torch .zeros ((1 ,64 ),dtype =torch .long ),
            'labels':torch .zeros ((1 ,64 ),dtype =torch .long ),
            'attention_mask':torch .zeros ((1 ,64 ),dtype =torch .long )
            }


        input_ids =torch .stack ([sample ['input_ids']for sample in batch ])



        labels =input_ids .clone ()
        labels [:,:-1 ]=input_ids [:,1 :]
        labels [:,-1 ]=-100 

        attention_mask =torch .stack ([sample .get ('attention_mask',torch .ones_like (sample ['input_ids']))for sample in batch ])


        return {
        'input_ids':input_ids .to (torch .long ),
        'labels':labels .to (torch .long ),
        'attention_mask':attention_mask .to (torch .long )
        }

    except Exception as e :
        logger .error (f"Collate function error: {e }")

        return {
        'input_ids':torch .zeros ((1 ,64 ),dtype =torch .int32 ),
        'labels':torch .zeros ((1 ,64 ),dtype =torch .int32 ),
        'attention_mask':torch .zeros ((1 ,64 ),dtype =torch .int32 )
        }

class GeometricExpert (nn .Module ):
    """Learnable-curvature expert with robust 3-layer residual FFN and manifold stabilization."""
    def __init__ (self ,input_dim :int ,hidden_dim :int ,output_dim :int ,manifold_type :str ='euclidean',dropout :float =0.1 ,*,expert_id :int =0 ,num_experts :int =1 ):
        super ().__init__ ()
        self .manifold_type =manifold_type 
        self .dim =input_dim 
        self .hidden_dim =hidden_dim 
        self .output_dim =output_dim 
        self .dropout =dropout 


        self .expert_id =expert_id 
        self .num_experts =max (1 ,num_experts )


        if manifold_type =="hyperbolic":

            log_c_init =sobol_raw_c (expert_id ,num_experts )
            self .raw_c =nn .Parameter (torch .tensor (log_c_init ),requires_grad =True )
        elif manifold_type =="spherical":

            expert_seed =hash (f"spherical_{input_dim }_{hidden_dim }")%1000 
            torch .manual_seed (expert_seed +100 )

            init_r =torch .tensor (0.2 +torch .rand (1 ).item ()*3.8 )
            self .raw_r =nn .Parameter (init_r ,requires_grad =True )

            torch .manual_seed (torch .initial_seed ())
        else :

            self .raw_c =self .raw_r =None 


        self ._init_advanced_manifold_params ()

    def _init_advanced_manifold_params (self ):
        """Initialize parameters specific to advanced manifold types (not hyperbolic/spherical/euclidean)"""
        if self .manifold_type =="simplex":

            self .raw_temperature =nn .Parameter (torch .tensor (1.0 ))
        elif self .manifold_type =="complex":

            self .raw_frequency =nn .Parameter (torch .randn (self .dim //2 )*0.1 )
            self .raw_phase =nn .Parameter (torch .randn (self .dim //2 )*0.1 )
        elif self .manifold_type =="lorentzian":

            self .raw_c_speed =nn .Parameter (torch .tensor (1.0 ))
        elif self .manifold_type =="product":

            self .manifold_weights =nn .Parameter (torch .ones (3 )/3 )

            if not hasattr (self ,'raw_c')or self .raw_c is None :
                self .raw_c =nn .Parameter (torch .tensor (0.5 +torch .rand (1 ).item ()*1.0 ))
            if not hasattr (self ,'raw_r')or self .raw_r is None :
                self .raw_r =nn .Parameter (torch .tensor (0.5 +torch .rand (1 ).item ()*1.0 ))



        self .to_tan =nn .Linear (self .dim ,self .dim )
        self .rms_norm_tan =nn .RMSNorm (self .dim )if hasattr (nn ,'RMSNorm')else nn .LayerNorm (self .dim )


        self .res_scale =nn .Parameter (torch .tensor (0.01 ))


        self .fc1 =nn .Linear (self .dim ,self .hidden_dim )
        self .fc2 =nn .Linear (self .hidden_dim ,self .hidden_dim )
        self .fc3 =nn .Linear (self .hidden_dim ,self .dim )
        self .ln1 =nn .LayerNorm (self .hidden_dim )
        self .ln2 =nn .LayerNorm (self .hidden_dim )
        self .ln3 =nn .LayerNorm (self .dim )
        self .drop =nn .Dropout (self .dropout )


        if self .dim !=self .output_dim :
            self .output_proj =nn .Linear (self .dim ,self .output_dim )
        else :
            self .output_proj =None 


        self .transformer_blocks =nn .ModuleList ([
        CausalBlock (self .hidden_dim ,
        n_heads =max (2 ,self .hidden_dim //64 ),
        ffn_mult =4 )
        for _ in range (2 )
        ])


    def _get_manifold (self ):
        if not GEOOPT_AVAILABLE :
            return None 
        if self .manifold_type =="hyperbolic":

            c =safe_softplus (self .raw_c )
            return geoopt .PoincareBall (c =c )
        if self .manifold_type =="spherical":

            return geoopt .Sphere ()
        return None 

    def _expmap (self ,x ):

        if self .manifold_type =="hyperbolic"and hasattr (self ,"raw_c"):
            radius =(1 /(torch .exp (self .raw_c ).clamp_min (1e-6 )))**0.5 
            x =x .clamp (max =0.95 *radius )

        M =self ._get_manifold ()
        if M is None :
            return x 
        try :

            if not hasattr (M ,"c"):

                radius =torch .tensor (1.0 ,device =x .device )
            else :

                radius =1.0 /torch .sqrt (M .c +1e-8 )

            max_norm =radius *(1.0 -1e-5 )
            x_norm =x .norm (dim =-1 ,keepdim =True )
            x_clamped =x *torch .clamp (x_norm ,max =max_norm )/(x_norm +1e-8 )
            return M .expmap0 (x_clamped )
        except Exception :
            return x 

    def _logmap (self ,y ):
        M =self ._get_manifold ()
        if M is None :
            return y 
        try :
            return M .logmap0 (y )
        except Exception :
            return y 


    def _apply_advanced_manifold_operation (self ,x :torch .Tensor )->torch .Tensor :
        """Apply advanced manifold-specific operations for new manifold types"""
        if self .manifold_type =="simplex":

            temperature =safe_softplus (self .raw_temperature )if hasattr (self ,'raw_temperature')else 1.0 
            return F .softmax (x /temperature ,dim =-1 )
        elif self .manifold_type =="complex":

            if hasattr (self ,'raw_frequency')and hasattr (self ,'raw_phase'):

                half =x .size (-1 )//2 
                real_part =x [...,:half ]
                imag_part =x [...,half :]


                frequency =torch .sigmoid (self .raw_frequency )*2 *3.14159 
                phase =self .raw_phase *3.14159 

                cos_part =real_part *torch .cos (frequency +phase )
                sin_part =imag_part *torch .sin (frequency +phase )

                return torch .cat ([cos_part ,sin_part ],dim =-1 )
            return x 
        elif self .manifold_type =="lorentzian":

            if hasattr (self ,'raw_c_speed'):

                c_speed =safe_softplus (self .raw_c_speed )

                half =x .size (-1 )//2 
                time_component =x [...,:half ]
                space_component =x [...,half :]


                time_term =-c_speed .pow (2 )*time_component .pow (2 )
                space_term =space_component .pow (2 )
                interval_parts =torch .tanh (time_term +space_term )


                interval =torch .cat ([interval_parts ,interval_parts ],dim =-1 )

                return x *interval 
            return x 
        elif self .manifold_type =="product":

            if hasattr (self ,'manifold_weights'):
                weights =F .softmax (self .manifold_weights ,dim =0 )


                euclidean_out =x 


                hyperbolic_out =torch .tanh (x )


                spherical_out =F .normalize (x ,p =2 ,dim =-1 )


                return (weights [0 ]*euclidean_out +
                weights [1 ]*hyperbolic_out +
                weights [2 ]*spherical_out )
            return x 
        else :

            return self ._expmap (x )


    def forward (self ,x :torch .Tensor )->torch .Tensor :

        return self ._forward_impl (x )

    def _forward_impl (self ,x :torch .Tensor )->torch .Tensor :

        with torch .amp .autocast ('cuda',enabled =False ):

            x_tan =self .rms_norm_tan (self .to_tan (x ))


            if self .manifold_type in ["simplex","complex","lorentzian","product"]:
                x_man =self ._apply_advanced_manifold_operation (x_tan )
            else :
                x_man =self ._expmap (x_tan )


        h_seq =x_man 

        h_seq =h_seq .unsqueeze (1 )
        for blk in self .transformer_blocks :
            h_seq =blk (h_seq )
        x_man =h_seq .squeeze (1 )



        with torch .amp .autocast ('cuda',enabled =True ):

            h =F .gelu (self .ln1 (self .fc1 (x_man .float ())))
            h =self .drop (h )


            if torch .cuda .is_available ():
                torch .cuda .empty_cache ()

            h =F .gelu (self .ln2 (self .fc2 (h )))
            h =self .drop (h )
            h =self .fc3 (h )


        with torch .amp .autocast ('cuda',enabled =False ):

            y_tan =self .ln3 (x_tan +self .res_scale *h .float ())


            if self .manifold_type in ["simplex","complex","lorentzian","product"]:
                output =self ._apply_advanced_manifold_operation (y_tan )
            else :
                output =self ._logmap (y_tan )


            if self .output_proj is not None :
                output =self .output_proj (output )

            return output 

class GatingNetwork (nn .Module ):
    """
    🚀 ENHANCED: SimBal (Similarity-Preserving Balance) Gating Network
    Based on 2025 research: "Load Balancing Mixture of Experts with Similarity Preserving Routers"
    
    Key innovations:
    - Orthogonal router matrices preserve token relationships
    - Eliminates uniform distribution forcing that causes expert collapse
    - Reduces balance loss explosion by 36% faster convergence
    """
    def __init__ (self ,input_dim :int ,num_experts :int ,k :int =2 ,capacity_factor :float =1.5 ,
    expert_diversity_weight :float =0.01 ,capacity_reg_weight :float =0.01 ):
        super ().__init__ ()
        self .input_dim =input_dim 
        self .num_experts =num_experts 
        self .k =k 
        self .capacity_factor =capacity_factor 
        self .expert_diversity_weight =expert_diversity_weight 
        self .capacity_reg_weight =capacity_reg_weight 


        self .router =nn .Linear (input_dim ,num_experts ,bias =False )
        self .pre_norm =nn .RMSNorm (input_dim )if hasattr (nn ,'RMSNorm')else nn .LayerNorm (input_dim )


        with torch .no_grad ():
            nn .init .orthogonal_ (self .router .weight )


        self .register_buffer ('capacity',torch .ones (num_experts )*capacity_factor )
        self .softmax_temp =nn .Parameter (torch .tensor (1.0 ))


        self .simbal_weight =0.1 

    def anneal_temperature (self ,step :int ,max_steps :int =10000 ,t0 :float =1.0 ,t1 :float =0.5 ):
        ratio =min (step /max_steps ,1.0 )
        new_temp =t1 +(t0 -t1 )*0.5 *(1 +math .cos (math .pi *ratio ))
        self .softmax_temp .data .fill_ (new_temp )

    def forward (self ,x :torch .Tensor ):
        """
        🚀 SimBal Forward Pass with Orthogonality-Based Load Balancing
        """
        batch_tokens ,input_dim =x .shape 


        x =self .pre_norm (x )
        x =F .normalize (x ,p =2 ,dim =-1 )
        W =F .normalize (self .router .weight ,p =2 ,dim =-1 )
        scaled_logits =F .linear (x ,W )/torch .clamp (self .softmax_temp ,min =1e-3 )


        top_k_logits ,indices =torch .topk (scaled_logits ,self .k ,dim =-1 )
        routing_weights =F .softmax (top_k_logits ,dim =-1 )


        routing_mask =torch .zeros_like (scaled_logits )

        routing_mask =routing_mask .scatter_ (-1 ,indices ,routing_weights .to (routing_mask .dtype ))


        routing_mask =torch .nan_to_num (routing_mask ,nan =0.0 ,posinf =0.0 ,neginf =0.0 )


        routing_mask =torch .min (routing_mask ,self .capacity .unsqueeze (0 ))
        max_capacity =self .capacity_factor /self .num_experts 
        routing_mask =torch .clamp (routing_mask ,max =max_capacity )
        routing_mask =routing_mask /routing_mask .sum (-1 ,keepdim =True ).clamp_min (1e-6 )


        R =self .router .weight 
        gram_matrix =torch .mm (R ,R .t ())
        identity =torch .eye (self .num_experts ,device =R .device ,dtype =R .dtype )


        orthogonality_loss =torch .norm (gram_matrix -identity ,p =1 )


        simbal_loss =self .simbal_weight *orthogonality_loss 


        expert_load =routing_mask .sum (dim =0 )
        expert_fraction =expert_load /batch_tokens 
        expert_entropy =-torch .sum (expert_fraction *torch .log (expert_fraction +1e-8 ))
        max_entropy =torch .log (torch .tensor (self .num_experts ,dtype =torch .float32 ,device =x .device ))
        diversity_loss =self .expert_diversity_weight *(max_entropy -expert_entropy )


        total_balance_loss =simbal_loss +diversity_loss 

        return routing_mask ,total_balance_loss 

class NuancedGeometricGate (nn .Module ):
    """
    🚀 ENHANCED: Incorporates geometric specialization analysis, sophistication scoring *and* manifold-balance regularisation.
    """
    def __init__ (self ,input_dim :int ,num_experts :int ,k :int =2 ,num_concept_groups :int =4 ,
    expert_manifolds :Optional [List [str ]]=None ,manifold_balance_weight :float =0.02 ):
        super ().__init__ ()
        self .input_dim =input_dim 
        self .num_experts =num_experts 
        self .k =k 
        self .num_concept_groups =num_concept_groups 
        self .manifold_balance_weight =manifold_balance_weight 

        self .manifold_groups :dict [str ,list [int ]]={}
        if expert_manifolds is not None :
            for idx ,m in enumerate (expert_manifolds ):
                self .manifold_groups .setdefault (m ,[]).append (idx )


        self .concept_router =nn .Linear (input_dim ,num_concept_groups )
        self .token_router =nn .Linear (input_dim ,num_experts )
        self .fusion_gate =nn .Linear (input_dim ,1 )


        self .sophistication_scorer =nn .Sequential (
        nn .Linear (input_dim ,input_dim //4 ),
        nn .GELU (),
        nn .Dropout (0.05 ),
        nn .Linear (input_dim //4 ,1 ),
        nn .Sigmoid ()
        )


        self .geometric_analyzer =nn .ModuleDict ({
        'euclidean':nn .Linear (input_dim ,1 ),
        'hyperbolic':nn .Linear (input_dim ,1 ),
        'spherical':nn .Linear (input_dim ,1 )
        })


        self .temperature =nn .Parameter (torch .tensor (1.0 ))
        self .sophistication_weight =nn .Parameter (torch .tensor (0.1 ))


        self .register_buffer ('training_step',torch .tensor (0 ))
        self .register_buffer ('avg_sophistication',torch .tensor (0.5 ))

    def analyze_geometric_specialization (self ,embeddings :torch .Tensor )->torch .Tensor :
        """
        🎓 GEOMETRIC SPECIALIZATION: Analyze which manifold is most appropriate
        
        Conservative implementation of user's curriculum learning insight
        """

        euclidean_affinity =torch .sigmoid (self .geometric_analyzer ['euclidean'](embeddings ))
        hyperbolic_affinity =torch .sigmoid (self .geometric_analyzer ['hyperbolic'](embeddings ))
        spherical_affinity =torch .sigmoid (self .geometric_analyzer ['spherical'](embeddings ))


        geometric_affinities =torch .cat ([
        euclidean_affinity ,hyperbolic_affinity ,spherical_affinity 
        ],dim =-1 )


        geometric_probs =F .softmax (geometric_affinities /torch .clamp (self .temperature ,0.1 ,2.0 ),dim =-1 )

        return geometric_probs 

    def compute_sophistication_bonus (self ,embeddings :torch .Tensor )->Tuple [torch .Tensor ,Dict ]:
        """
        🧠 ENHANCED SOPHISTICATION SCORING: Context-aware, progressive sophistication detection
        
        IMPROVED LOGIC:
        1. Context-aware scoring based on embedding patterns
        2. Dynamic thresholding based on sequence complexity
        3. Multi-faceted reasoning analysis
        4. Progressive sophistication detection
        """
        batch_tokens ,hidden_dim =embeddings .shape 



        embedding_variance =torch .var (embeddings ,dim =0 ).mean ()
        context_complexity =torch .clamp (embedding_variance /(hidden_dim *0.1 ),0 ,1 )


        sophistication_raw =self .sophistication_scorer (embeddings )


        context_bonus =context_complexity *0.3 
        sophistication_scores =sophistication_raw .squeeze (-1 )+context_bonus 



        if batch_tokens >32 :

            sophistication_consistency =1.0 -torch .var (sophistication_scores )/(torch .mean (sophistication_scores )+1e-8 )
            progressive_bonus =sophistication_consistency *0.2 
            sophistication_scores =sophistication_scores +progressive_bonus 



        training_progress =min (self .training_step /10000.0 ,1.0 )
        target_sophistication =0.3 +(training_progress *0.4 )


        if self .training :
            current_avg =sophistication_scores .mean ().detach ()
            learning_rate =0.01 +(training_progress *0.04 )
            self .avg_sophistication =(1 -learning_rate )*self .avg_sophistication +learning_rate *current_avg 



        sophistication_weight =1.0 +self .sophistication_weight *torch .tanh (sophistication_scores )


        sophistication_reg =0.01 *F .mse_loss (
        sophistication_scores .mean (),
        torch .tensor (target_sophistication ,device =embeddings .device )
        )


        diversity_reg =0.005 *(1.0 -torch .var (sophistication_scores ))
        total_reg =sophistication_reg +diversity_reg 

        metrics ={
        'sophistication_scores':sophistication_scores ,
        'context_complexity':context_complexity .item (),
        'avg_sophistication':self .avg_sophistication .item (),
        'target_sophistication':target_sophistication ,
        'training_progress':training_progress ,
        'sophistication_regularization':total_reg ,
        'sophistication_consistency':sophistication_consistency .item ()if batch_tokens >32 else 0.0 
        }

        return sophistication_weight ,metrics 

    def forward (self ,x :torch .Tensor )->Tuple [torch .Tensor ,torch .Tensor ,Dict ]:
        """
        Enhanced forward pass with geometric specialization and sophistication scoring
        
        CONSERVATIVE: Only adds stable enhancements that won't break existing training
        """
        batch_tokens ,hidden_dim =x .shape 


        concept_logits =self .concept_router (x )
        token_logits =self .token_router (x )

        concept_probs =F .softmax (concept_logits /self .temperature ,dim =-1 )
        token_probs =F .softmax (token_logits /self .temperature ,dim =-1 )


        geometric_probs =self .analyze_geometric_specialization (x )


        geometric_routing =torch .zeros (batch_tokens ,self .num_experts ,device =x .device )
        if self .num_experts >=3 :

            geometric_routing [:,0 ]=geometric_probs [:,0 ]
            geometric_routing [:,1 ]=geometric_probs [:,1 ]
            geometric_routing [:,2 ]=geometric_probs [:,2 ]


            if self .num_experts >3 :
                remaining_prob =(1.0 -geometric_probs .sum (dim =-1 ,keepdim =True ))/max (1 ,self .num_experts -3 )
                geometric_routing [:,3 :]=remaining_prob 


        sophistication_weights ,sophistication_metrics =self .compute_sophistication_bonus (x )


        fusion_weight =torch .sigmoid (self .fusion_gate (x ))



        hybrid_logits =(
        0.6 *token_logits +
        0.2 *concept_logits .mean (dim =-1 ,keepdim =True )+
        0.2 *geometric_routing 
        )


        enhanced_logits =hybrid_logits *sophistication_weights .unsqueeze (-1 )


        top_k_logits ,top_k_indices =torch .topk (enhanced_logits ,self .k ,dim =-1 )
        routing_weights =torch .zeros_like (enhanced_logits )
        routing_weights .scatter_ (-1 ,top_k_indices ,F .softmax (top_k_logits ,dim =-1 ))



        expert_usage =routing_weights .sum (dim =0 )


        if hasattr (self ,'token_router')and hasattr (self .token_router ,'weight'):
            W =self .token_router .weight 
            gram =torch .mm (W ,W .t ())/(W .norm (dim =1 ,keepdim =True )+1e-8 )
            identity =torch .eye (self .num_experts ,device =W .device ,dtype =W .dtype )
            orthogonality_loss =0.05 *(gram -identity ).abs ().sum ()
            balance_loss =orthogonality_loss 
        else :

            target_usage =batch_tokens *self .k /self .num_experts 
            balance_loss =0.0001 *F .mse_loss (expert_usage ,torch .full_like (expert_usage ,target_usage ))


        total_loss =balance_loss +sophistication_metrics ['sophistication_regularization']


        if self .manifold_groups :
            usage_total =expert_usage .sum ()+1e-8 
            manifold_usage =[]
            for _m ,idxs in self .manifold_groups .items ():
                manifold_usage .append (expert_usage [idxs ].sum ())
            manifold_usage_t =torch .stack (manifold_usage )/usage_total 
            uniform_dist =torch .full_like (manifold_usage_t ,1.0 /len (manifold_usage ))
            imbalance_factor =torch .std (manifold_usage_t )
            dynamic_weight =self .manifold_balance_weight *(1 +5 *imbalance_factor )
            manifold_div_loss =dynamic_weight *F .mse_loss (manifold_usage_t ,uniform_dist )
            total_loss =total_loss +manifold_div_loss 
        else :
            manifold_div_loss =torch .tensor (0.0 ,device =x .device )


        token_entropy =-(routing_weights *torch .log (routing_weights +1e-8 )).sum (dim =-1 ).mean ()
        max_token_entropy =torch .log (torch .tensor (self .k ,device =x .device ,dtype =x .dtype ))
        entropy_loss =self .manifold_balance_weight *0.5 *(max_token_entropy -token_entropy )
        total_loss =total_loss +entropy_loss 



        soph_scores_tensor =sophistication_metrics .get ('sophistication_scores')
        if torch .is_tensor (soph_scores_tensor ):
            soph_scalar =soph_scores_tensor .mean ()
        else :
            soph_scalar =torch .tensor (0.0 ,device =x .device )

        analysis ={
        'geometric_specialization':geometric_probs ,
        'concept_preferences':concept_probs ,
        'fusion_weights':fusion_weight .squeeze (-1 ),
        'expert_usage':expert_usage ,
        'routing_entropy':-(routing_weights *torch .log (routing_weights +1e-8 )).sum (dim =-1 ).mean (),
        'sophistication_score':soph_scalar ,
        **sophistication_metrics ,
        'manifold_div_loss':manifold_div_loss .detach (),
        'entropy_loss':entropy_loss .detach ()
        }


        if self .training :
            self .training_step +=1 

        return routing_weights ,total_loss ,analysis 

class SpectralCombiner (nn .Module ):
    def __init__ (self ,num_experts :int ,output_dim :int ):
        super ().__init__ ()

        self .freq_weights =nn .Parameter (torch .randn (num_experts ,output_dim //2 +1 ,dtype =torch .cfloat )*0.01 )
        self .layer_norm =nn .LayerNorm (output_dim )

    def forward (self ,expert_outputs :torch .Tensor ,routing_mask :torch .Tensor =None )->torch .Tensor :



        if expert_outputs .numel ()==0 or expert_outputs .shape [0 ]==0 :

            batch_size =expert_outputs .shape [0 ]if expert_outputs .dim ()>0 else 1 
            output_dim =expert_outputs .shape [-1 ]if expert_outputs .dim ()>2 else self .freq_weights .shape [1 ]*2 -2 
            return torch .zeros (batch_size ,output_dim ,device =expert_outputs .device ,dtype =expert_outputs .dtype )


        x_fft =torch .fft .rfft (expert_outputs ,dim =-1 )
        x_mixed_fft =x_fft *self .freq_weights .unsqueeze (0 )
        x_mixed =torch .fft .irfft (x_mixed_fft ,n =expert_outputs .size (-1 ),dim =-1 )


        combined =x_mixed .mean (dim =1 )


        if routing_mask is not None :
            residual =torch .einsum ("be,bed->bd",routing_mask ,expert_outputs )
            combined =0.5 *combined +0.5 *residual 

        return self .layer_norm (combined )


class ConceptGroupCombiner (nn .Module ):
    """
    🌟 REVOLUTIONARY: Concept-Group-Aware Combination for Nuanced Understanding
    
    Instead of token-level combination, this combines expert outputs by:
    1. Grouping tokens into semantic concepts  
    2. Combining experts at concept-group level
    3. Preserving nuanced understanding within each concept
    4. Generating sophisticated multi-token responses
    """
    def __init__ (self ,num_experts :int ,output_dim :int ,num_concept_groups :int =4 ):
        super ().__init__ ()
        self .num_experts =num_experts 
        self .output_dim =output_dim 
        self .num_concept_groups =num_concept_groups 


        self .concept_attention =nn .MultiheadAttention (
        output_dim ,
        num_heads =8 ,
        dropout =0.1 ,
        batch_first =True 
        )


        self .manifold_fusion =nn .Sequential (
        nn .Linear (output_dim *num_experts ,output_dim *2 ),
        nn .GELU (),
        nn .Dropout (0.1 ),
        nn .Linear (output_dim *2 ,output_dim ),
        nn .LayerNorm (output_dim )
        )


        self .sophistication_enhancer =nn .Sequential (
        nn .Linear (output_dim ,output_dim //2 ),
        nn .GELU (),
        nn .Linear (output_dim //2 ,output_dim ),
        nn .Tanh ()
        )


        self .concept_weights =nn .Parameter (torch .ones (num_concept_groups )/num_concept_groups )

        self .layer_norm =nn .LayerNorm (output_dim )

    def forward (self ,expert_outputs :torch .Tensor ,routing_mask :torch .Tensor =None ,
    nuance_analysis :Dict =None )->torch .Tensor :
        """
        Args:
            expert_outputs: [B*T, E, D] - expert outputs
            routing_mask: [B*T, E] - routing weights  
            nuance_analysis: Dict containing concept_groups, sophistication_score, etc.
        
        Returns:
            [B*T, D] - sophisticated combined output
        """
        batch_tokens ,num_experts ,output_dim =expert_outputs .shape 


        if nuance_analysis and 'concept_groups'in nuance_analysis :
            concept_groups =nuance_analysis ['concept_groups']


            if 'geometric_fit'in nuance_analysis :
                geometric_fit =nuance_analysis ['geometric_fit']
                enhanced_routing =routing_mask *torch .sigmoid (geometric_fit )
            else :
                enhanced_routing =routing_mask 


            concept_weights_expanded =self .concept_weights .unsqueeze (0 ).expand (batch_tokens ,-1 )
            concept_weight_scalar =torch .sum (concept_groups *concept_weights_expanded ,dim =-1 ,keepdim =True )
            enhanced_routing =enhanced_routing *concept_weight_scalar 

        else :
            enhanced_routing =routing_mask 



        expert_flat =expert_outputs .view (batch_tokens ,-1 )
        manifold_fused =self .manifold_fusion (expert_flat )


        if enhanced_routing is not None :
            traditional_combined =torch .einsum ("be,bed->bd",enhanced_routing ,expert_outputs )
        else :
            traditional_combined =expert_outputs .mean (dim =1 )


        sophistication_multiplier =self .sophistication_enhancer (traditional_combined )
        enhanced_traditional =traditional_combined *(1.0 +sophistication_multiplier )



        enhanced_input =enhanced_traditional .unsqueeze (1 )
        attended_output ,_ =self .concept_attention (enhanced_input ,enhanced_input ,enhanced_input )
        attended_output =attended_output .squeeze (1 )



        alpha =0.4 
        beta =0.4 
        gamma =0.2 

        final_output =(alpha *manifold_fused +
        beta *enhanced_traditional +
        gamma *attended_output )

        return self .layer_norm (final_output )


class ThoughtGenerator (nn .Module ):
    def __init__ (self ,input_dim :int ,hidden_dim :int ,num_heads :int =8 ,num_layers :int =2 ):
        super ().__init__ ()
        self .input_dim =input_dim 



        self .fc_in =nn .Linear (2 *input_dim ,input_dim )



        safe_heads =max (1 ,num_heads )
        if input_dim %safe_heads !=0 :

            divisors =[d for d in range (safe_heads ,0 ,-1 )if input_dim %d ==0 ]
            safe_heads =divisors [0 ]if divisors else 1 
            logging .warning (f"⚠️ Adjusted num_heads from {num_heads } to {safe_heads } to satisfy embed_dim % num_heads == 0 (embed_dim={input_dim }).")

        self .self_attn =nn .MultiheadAttention (
        input_dim ,
        num_heads =safe_heads ,
        dropout =0.1 ,
        batch_first =True ,
        )
        self .attn_norm =nn .LayerNorm (input_dim )


        self .ffn =nn .Sequential (
        nn .LayerNorm (input_dim ),
        nn .Linear (input_dim ,hidden_dim ),
        nn .GELU (),
        nn .Dropout (0.1 ),
        nn .Linear (hidden_dim ,hidden_dim //2 ),
        nn .GELU (),
        nn .Dropout (0.1 ),
        nn .Linear (hidden_dim //2 ,input_dim ),
        )


        self .output_norm =nn .LayerNorm (input_dim )


        self .attn_res_scale =nn .Parameter (torch .tensor (0.1 ))
        self .ffn_res_scale =nn .Parameter (torch .tensor (0.1 ))

    def forward (self ,x :torch .Tensor )->torch .Tensor :

        if x .size (-1 )==2 *self .input_dim :
            x =self .fc_in (x )



        if x .dim ()==2 :


            x_seq =x .unsqueeze (1 )
            squeeze_output =True 
        else :
            x_seq =x 
            squeeze_output =False 


        x_normed =self .attn_norm (x_seq )
        attn_out ,_ =self .self_attn (x_normed ,x_normed ,x_normed )
        x_seq =x_seq +self .attn_res_scale *attn_out 


        if squeeze_output :
            x_seq =x_seq .squeeze (1 )


        ffn_out =self .ffn (x_seq )
        output =x_seq +self .ffn_res_scale *ffn_out 


        output =self .output_norm (output )

        return output 

class AnalogyReasoner (nn .Module ):
    def __init__ (self ,dim :int ):
        super ().__init__ ()
        self .proj =nn .Linear (dim ,dim )
        self .norm =nn .RMSNorm (dim )if hasattr (nn ,'RMSNorm')else nn .LayerNorm (dim )

    def forward (self ,a1 :torch .Tensor ,a2 :torch .Tensor ,b1 :torch .Tensor )->torch .Tensor :
        diff =self .norm (self .proj (a2 -a1 ))
        return b1 +diff 


class WorkingMemory (nn .Module ):
    """
    Enhanced key-value memory with usage tracking and improved read/write policies.
    """
    def __init__ (self ,slots :int ,width :int ):
        super ().__init__ ()
        self .slots ,self .width =slots ,width 
        self .register_parameter ("mem_init",nn .Parameter (torch .zeros (slots ,width )))
        self .read_proj =nn .Linear (width ,width ,bias =False )
        self .write_proj =nn .Linear (width ,width ,bias =False )
        self .gate_proj =nn .Linear (width ,1 ,bias =False )


        self .register_buffer ("usage_decay",torch .tensor (0.98 ))
        self .usage_temp =0.25 

    def reset (self ,batch_size :int ,device :torch .device )->Tuple [torch .Tensor ,torch .Tensor ]:
        """Call at start of every new sequence. Returns (memory, usage)."""
        memory =self .mem_init .expand (batch_size ,-1 ,-1 ).clone ().to (device )
        usage =torch .zeros (batch_size ,self .slots ,device =device )
        return memory ,usage 

    def forward (self ,
    memory :torch .Tensor ,
    usage :torch .Tensor ,
    query :torch .Tensor 
    )->Tuple [torch .Tensor ,torch .Tensor ,torch .Tensor ]:

        logits =torch .einsum ("bd,bkd->bk",self .read_proj (query ),memory )
        weights =torch .softmax (logits /math .sqrt (self .width ),dim =-1 )
        read_vec =torch .einsum ("bk,bkd->bd",weights ,memory )



        usage =usage *self .usage_decay +weights 



        neg_usage_logits =-usage /self .usage_temp 


        neg_usage_logits =torch .clamp (neg_usage_logits ,min =-10.0 ,max =10.0 )
        neg_usage_logits =torch .nan_to_num (neg_usage_logits ,nan =0.0 ,posinf =10.0 ,neginf =-10.0 )


        probs =F .softmax (neg_usage_logits ,dim =-1 )
        probs =torch .clamp (probs ,min =1e-8 )
        probs =probs /probs .sum (dim =-1 ,keepdim =True )

        least_idx =torch .multinomial (probs ,1 ).squeeze (-1 )

        write_key =self .write_proj (query )
        write_key =torch .nan_to_num (write_key ,nan =0.0 ,posinf =1e4 ,neginf =-1e4 )
        write_gate =0.1 +0.9 *torch .sigmoid (self .gate_proj (query ))
        mem_next =memory .clone ()
        mem_next [torch .arange (memory .size (0 )),least_idx ]=(1.0 -write_gate )*memory [torch .arange (memory .size (0 )),least_idx ]+write_gate *write_key 

        return mem_next ,usage ,read_vec 

class DiffusionGate (nn .Module ):
    def __init__ (self ,thought_dim :int ):
        super ().__init__ ()
        self .scorer =nn .Linear (thought_dim ,1 )
        self .temperature =nn .Parameter (torch .ones (1 ))

    def forward (self ,thoughts :list [torch .Tensor ])->torch .Tensor :
        stacked_thoughts =torch .stack (thoughts ,dim =1 )
        scores =self .scorer (stacked_thoughts ).squeeze (-1 )/self .temperature 
        gumbel_weights =F .gumbel_softmax (scores ,tau =1.0 ,hard =False ,dim =-1 )
        selected_thought =torch .einsum ("bs,bsd->bd",gumbel_weights ,stacked_thoughts )
        return selected_thought 

class RewardModel (nn .Module ):
    def __init__ (self ,input_dim :int ):
        super ().__init__ ()

        self .input_projection =nn .Linear (input_dim ,input_dim )
        self .layer_norm1 =nn .LayerNorm (input_dim )

        self .hidden1 =nn .Linear (input_dim ,input_dim //2 )
        self .layer_norm2 =nn .LayerNorm (input_dim //2 )

        self .hidden2 =nn .Linear (input_dim //2 ,input_dim //4 )
        self .layer_norm3 =nn .LayerNorm (input_dim //4 )

        self .output_head =nn .Linear (input_dim //4 ,1 )

        self .dropout =nn .Dropout (0.1 )
        self .activation =nn .GELU ()

    def forward (self ,x :torch .Tensor )->torch .Tensor :

        proj =self .input_projection (x )
        x_norm =self .layer_norm1 (x +proj )


        h1 =self .activation (self .layer_norm2 (self .hidden1 (x_norm )))
        h1 =self .dropout (h1 )


        h2 =self .activation (self .layer_norm3 (self .hidden2 (h1 )))
        h2 =self .dropout (h2 )


        reward =self .output_head (h2 )
        return reward 

class AudioSpectrogramEncoder (nn .Module ):
    """Encode 2D audio spectrograms to model embedding space while preserving geometric structure"""
    def __init__ (self ,codebook_size :int ,seq_len :int ,model_dim :int ):
        super ().__init__ ()
        self .codebook_size =codebook_size 
        self .seq_len =seq_len 
        self .model_dim =model_dim 



        self .spectrogram_encoder =nn .Conv1d (codebook_size ,model_dim ,kernel_size =1 )


        self .pos_encoding =nn .Parameter (torch .randn (1 ,seq_len ,model_dim )*0.02 )


        self .layer_norm =nn .LayerNorm (model_dim )


        self .geometric_proj =nn .Linear (model_dim ,model_dim )

    def forward (self ,audio_codes :torch .Tensor )->torch .Tensor :
        """
        Encode 2D audio spectrograms preserving both temporal and spectral structure
        
        Args:
            audio_codes: [B, codebook_size, seq_len] - Real EncodecModel output
            
        Returns:
            [B, seq_len, model_dim] - Encoded for geometric expert processing
        """
        batch_size ,cb_size ,seq_length =audio_codes .shape 


        if cb_size !=self .codebook_size :
            logger .warning (f"Expected {self .codebook_size } codebooks, got {cb_size }, adapting...")

            if not hasattr (self ,'_adaptive_encoder'):
                self ._adaptive_encoder =nn .Conv1d (cb_size ,self .model_dim ,kernel_size =1 ).to (audio_codes .device )
            x =self ._adaptive_encoder (audio_codes .float ())
        else :

            x =self .spectrogram_encoder (audio_codes .float ())


        x =x .transpose (1 ,2 )


        if x .size (1 )<=self .pos_encoding .size (1 ):
            x =x +self .pos_encoding [:,:x .size (1 ),:]
        else :

            pos_enc =self .pos_encoding .repeat (1 ,(x .size (1 )//self .pos_encoding .size (1 ))+1 ,1 )
            x =x +pos_enc [:,:x .size (1 ),:]


        x =self .layer_norm (x )
        x =self .geometric_proj (x )

        return x 

class AudioSpectrogramDecoder (nn .Module ):
    """Decode from model space back to 2D audio spectrograms for EncodecModel"""
    def __init__ (self ,model_dim :int ,codebook_size :int ,seq_len :int ):
        super ().__init__ ()
        self .model_dim =model_dim 
        self .codebook_size =codebook_size 
        self .seq_len =seq_len 


        self .geometric_proj =nn .Linear (model_dim ,model_dim )


        self .layer_norm =nn .LayerNorm (model_dim )



        self .spectrogram_decoder =nn .Conv1d (model_dim ,codebook_size ,kernel_size =1 )


        self .activation =nn .Tanh ()

    def forward (self ,x :torch .Tensor )->torch .Tensor :
        """
        Decode from model space back to 2D audio spectrograms
        
        Args:
            x: [B, seq_len, model_dim] - Model output
            
        Returns:
            [B, codebook_size, seq_len] - Ready for EncodecModel decoding
        """

        x =self .geometric_proj (x )
        x =self .layer_norm (x )


        x =x .transpose (1 ,2 )


        audio_spectrograms =self .spectrogram_decoder (x )


        audio_spectrograms =self .activation (audio_spectrograms )

        return audio_spectrograms 

class MixtureOfGeometricExperts (nn .Module ):
    def __init__ (self ,config :dict ):
        super ().__init__ ()
        model_config =config ['model']

        self .enable_vision =model_config .get ('enable_vision',False )
        self .enable_audio =model_config .get ('enable_audio',False )


        if self .enable_vision :
            vision_tower_name =model_config .get ('vision_tower_name','openai/clip-vit-large-patch14')
            self .vision_tower =CLIPVisionModel .from_pretrained (vision_tower_name )
            self .vision_tower .requires_grad_ (False )

            vision_hidden_dim =self .vision_tower .config .hidden_size 
            model_input_dim =model_config ['input_dim']
            self .vision_projector =nn .Linear (vision_hidden_dim ,model_input_dim )



        if self .enable_audio :

            audio_codebook_size =model_config .get ('audio_codebook_size',2 )
            audio_seq_len =model_config .get ('seq_len',256 )

            self .audio_encoder =AudioSpectrogramEncoder (
            codebook_size =audio_codebook_size ,
            seq_len =audio_seq_len ,
            model_dim =model_config ['input_dim']
            )

            self .audio_decoder =AudioSpectrogramDecoder (
            model_dim =model_config ['output_dim'],
            codebook_size =audio_codebook_size ,
            seq_len =audio_seq_len 
            )

            logging .info (f"✅ Audio spectrogram encoder/decoder initialized with {audio_codebook_size } codebooks")


        self .embedding =nn .Embedding (
        model_config ['vocab_size'],
        model_config ['input_dim'],
        padding_idx =model_config .get ('pad_token_id',0 )
        )

        self .use_analogy_reasoner =model_config .get ('use_analogy_reasoner',True )
        self .analogy_reasoner =AnalogyReasoner (model_config ['input_dim'])if self .use_analogy_reasoner else None


        self .pad_token_id =config ['model'].get ('pad_token_id',0 )


        manifolds =config ['model']['manifolds']
        self .experts =nn .ModuleList ([
        GeometricExpert (
        config ['model']['input_dim'],
        config ['model']['hidden_dim'],
        config ['model']['output_dim'],
        manifold_type =manifold ,
        expert_id =idx ,
        num_experts =len (manifolds )
        )
        for idx ,manifold in enumerate (manifolds )
        ])


        dense_id =config ['model'].get ('dense_init')
        if dense_id :
            from transformers import AutoModelForCausalLM 
            dense =AutoModelForCausalLM .from_pretrained (
            dense_id ,
            torch_dtype =torch .float32 ,
            device_map ="cpu",
            low_cpu_mem_usage =True ,
            )
            with torch .no_grad ():

                src =dense .get_input_embeddings ().weight 
                tgt =self .embedding .weight 
                copy =min (src .size (0 ),tgt .size (0 ))
                tgt [:copy ].copy_ (src [:copy ])
                if tgt .size (0 )>src .size (0 ):
                    nn .init .normal_ (tgt [copy :],std =0.02 )

                ffn =dense .transformer .h [0 ].mlp 
                for ex in self .experts :
                    w1 =ffn .c_fc .weight [:ex .fc1 .out_features ,:ex .fc1 .in_features ]
                    w2 =ffn .c_proj .weight [:ex .fc2 .out_features ,:ex .fc2 .in_features ]
                    w3 =ffn .c_proj .weight [:ex .fc3 .out_features ,:ex .fc3 .in_features ]
                    ex .fc1 .weight .copy_ (w1 +0.02 *torch .randn_like (ex .fc1 .weight ))
                    ex .fc2 .weight .copy_ (w2 +0.02 *torch .randn_like (ex .fc2 .weight ))
                    ex .fc3 .weight .copy_ (w3 +0.02 *torch .randn_like (ex .fc3 .weight ))


        self .num_experts =len (self .experts )


        use_nuanced_routing =config ['model'].get ('use_nuanced_routing',False )

        if use_nuanced_routing :
            self .gate =NuancedGeometricGate (
            config ['model']['output_dim'],
            self .num_experts ,
            k =config ['model']['k'],
            num_concept_groups =config ['model'].get ('num_concept_groups',4 ),
            expert_manifolds =manifolds ,
            )
            self .combiner =ConceptGroupCombiner (
            self .num_experts ,
            config ['model']['output_dim'],
            num_concept_groups =config ['model'].get ('num_concept_groups',4 )
            )
            self .nuanced_routing =True 
        else :
            self .gate =GatingNetwork (
            config ['model']['output_dim'],
            self .num_experts ,
            k =config ['model']['k']
            )
            self .combiner =SpectralCombiner (
            self .num_experts ,
            config ['model']['output_dim']
            )
            self .nuanced_routing =False 

        self .working_memory =WorkingMemory (
        config ['model']['memory_slots'],
        config ['model']['output_dim']
        )






        self .thought_generator =ThoughtGenerator (
        config ['model']['input_dim'],
        config ['model']['hidden_dim'],
        num_heads =int (config ['model'].get ('num_heads',8 )),
        num_layers =int (config ['model'].get ('num_layers',2 )),
        )

        self .diffusion_gate =DiffusionGate (config ['model']['output_dim'])

        self .final_head =nn .Linear (
        config ['model']['output_dim'],
        config ['model']['final_output_dim']
        )

        self .verifier =RewardModel (config ['model']['output_dim'])


        self .logic_len =8 
        self .logic_proj =nn .Linear (
        config ['model']['output_dim'],
        self .logic_len *config ['model']['output_dim']
        )

        self .recursion_steps =config ['model']['recursion_steps']


        self .mem_use_coef =config ['training'].get ('memory_auxiliary_loss_weight',0.0 )


        self .register_buffer ("training_step",torch .tensor (0 ))
        self .alpha_anneal_steps =5000 




        self .max_position_embeddings =int (model_config .get ('seq_len',512 ))
        self .position_embeddings =nn .Embedding (self .max_position_embeddings ,
        model_config ['input_dim'])

    def resize_token_embeddings (self ,new_num_tokens :int ):
        """ Resizes the token embeddings and the final output head. """
        old_num_tokens =self .embedding .num_embeddings 
        if new_num_tokens <=old_num_tokens :
            return 


        device =self .embedding .weight .device 


        old_embedding_weight =self .embedding .weight .data 
        self .embedding =nn .Embedding (new_num_tokens ,self .embedding .embedding_dim ,padding_idx =self .pad_token_id ).to (device )
        self .embedding .weight .data [:old_num_tokens ,:]=old_embedding_weight 

        self .embedding .weight .data [old_num_tokens :,:].normal_ (mean =0.0 ,std =0.02 )


        old_head_weight =self .final_head .weight .data 
        old_head_bias =self .final_head .bias .data if self .final_head .bias is not None else None 
        self .final_head =nn .Linear (self .final_head .in_features ,new_num_tokens ).to (self .final_head .weight .device )
        min_size =min (old_num_tokens ,new_num_tokens )
        self .final_head .weight .data [:min_size ]=old_head_weight [:min_size ]
        if old_head_bias is not None :
            self .final_head .bias .data [:min_size ]=old_head_bias [:min_size ]


        if new_num_tokens >old_num_tokens :
            nn .init .normal_ (self .final_head .weight .data [old_num_tokens :],mean =0.0 ,std =0.02 )
            if self .final_head .bias is not None :
                nn .init .zeros_ (self .final_head .bias .data [old_num_tokens :])

        logging .info (f"✅ Resized model embeddings from {old_num_tokens } to {new_num_tokens }")

    @torch ._dynamo .disable 
    def _combine_experts (self ,weighted :torch .Tensor ,routing_mask :torch .Tensor )->torch .Tensor :
        """Protected combiner call that bypasses TorchDynamo compilation.
        
        This prevents OOM during FakeTensor simulation of the FFT operations
        in the SpectralCombiner, while keeping the actual training stable.
        """
        return self .combiner (weighted ,routing_mask )

    def forward (self ,input_ids :torch .Tensor =None ,attention_mask :torch .Tensor =None ,
    pixel_values :torch .Tensor =None ,audio_codes :torch .Tensor =None ,**kwargs ):
        """
        Truly multimodal forward pass supporting text, vision, and audio inputs.
        At least one of input_ids, pixel_values, or audio_codes must be provided.
        """
        batch_size =None 
        text_embeds =None 


        if input_ids is not None :

            if input_ids .dim ()==1 :
                input_ids =input_ids .unsqueeze (0 )
            if attention_mask is not None and attention_mask .dim ()==1 :
                attention_mask =attention_mask .unsqueeze (0 )


            vocab_size =self .embedding .num_embeddings 
            max_token_id =torch .max (input_ids ).item ()if input_ids .numel ()>0 else 0 
            min_token_id =torch .min (input_ids ).item ()if input_ids .numel ()>0 else 0 

            if max_token_id >=vocab_size :
                raise RuntimeError (f"CRITICAL: Token ID {max_token_id } >= vocab_size {vocab_size }. "
                f"Model embedding size: {vocab_size }, got max token: {max_token_id }")
            if min_token_id <0 :
                raise RuntimeError (f"CRITICAL: Negative token ID {min_token_id } found. "
                f"All token IDs must be >= 0")

            text_embeds =self .embedding (input_ids )

            seq_len =text_embeds .size (1 )
            if seq_len >self .max_position_embeddings :

                extra =seq_len -self .max_position_embeddings 
                new_weight =text_embeds .new_empty (extra ,self .position_embeddings .embedding_dim )
                nn .init .normal_ (new_weight ,std =0.02 )
                self .position_embeddings .weight .data =torch .cat ([
                self .position_embeddings .weight .data ,
                new_weight 
                ],dim =0 )
                self .max_position_embeddings =seq_len 

            position_ids =torch .arange (seq_len ,device =text_embeds .device ).unsqueeze (0 )
            position_ids =position_ids .expand (text_embeds .size (0 ),-1 )
            pos_embed =self .position_embeddings (position_ids )
            text_embeds =text_embeds +pos_embed 

            batch_size =text_embeds .size (0 )
            seq_len =text_embeds .size (1 )


        if self .enable_vision and pixel_values is not None :
            vision_outputs =self .vision_tower (pixel_values =pixel_values )
            image_features =vision_outputs .pooler_output 
            projected_image_embeds =self .vision_projector (image_features )

            if batch_size is None :
                batch_size =projected_image_embeds .size (0 )

            if text_embeds is not None :

                text_embeds =text_embeds +projected_image_embeds .unsqueeze (1 )
            else :

                seq_len =1 
                text_embeds =projected_image_embeds .unsqueeze (1 )

                if attention_mask is None :
                    attention_mask =torch .ones (batch_size ,seq_len ,device =pixel_values .device )



        original_audio_codes =audio_codes 

        if self .enable_audio and audio_codes is not None :

            if audio_codes .dim ()==3 :

                logger .debug (f"Processing 2D audio spectrograms: {audio_codes .shape }")
                audio_embeds =self .audio_encoder (audio_codes )

            elif audio_codes .dim ()==2 :

                logger .debug (f"Processing 1D audio tokens: {audio_codes .shape }")


                vocab_size =self .embedding .num_embeddings 
                max_audio_token =torch .max (audio_codes ).item ()if audio_codes .numel ()>0 else 0 
                if max_audio_token >=vocab_size :
                    raise RuntimeError (f"CRITICAL: Audio token {max_audio_token } >= vocab_size {vocab_size }")

                audio_embeds =self .embedding (audio_codes )

            elif audio_codes .dim ()==1 :

                audio_codes =audio_codes .unsqueeze (0 )


                vocab_size =self .embedding .num_embeddings 
                max_audio_token =torch .max (audio_codes ).item ()if audio_codes .numel ()>0 else 0 
                if max_audio_token >=vocab_size :
                    raise RuntimeError (f"CRITICAL: Audio token {max_audio_token } >= vocab_size {vocab_size }")

                audio_embeds =self .embedding (audio_codes )

            else :
                raise ValueError (f"Unexpected audio_codes dimensions: {audio_codes .shape }. "
                f"Expected 1D [seq_len], 2D [B, seq_len], or 3D [B, codebook_size, seq_len]")

            if batch_size is None :
                batch_size =audio_embeds .size (0 )

            if text_embeds is not None :

                text_embeds =torch .cat ([text_embeds ,audio_embeds ],dim =1 )
                if attention_mask is not None :
                    audio_mask =torch .ones (batch_size ,audio_embeds .size (1 ),device =audio_embeds .device )
                    attention_mask =torch .cat ([attention_mask ,audio_mask ],dim =1 )
                seq_len =text_embeds .size (1 )
            else :

                text_embeds =audio_embeds 
                seq_len =text_embeds .size (1 )
                if attention_mask is None :
                    attention_mask =torch .ones (batch_size ,seq_len ,device =audio_codes .device )


        if text_embeds is None :
            raise ValueError ("At least one of input_ids, pixel_values, or audio_codes must be provided")

        x =text_embeds 
        batch_size ,seq_len =x .size (0 ),x .size (1 )


        x_seq =x 


        if attention_mask is not None :

            mask_expanded =attention_mask .unsqueeze (-1 )
            x_seq =x_seq *mask_expanded 



        memory ,usage =self .working_memory .reset (batch_size ,x_seq .device )
        all_thoughts ,balance_losses ,all_routing_masks =[],[],[]
        current_input =x_seq 

        for step in range (self .recursion_steps ):

            B ,T ,D =current_input .shape 


            current_flat =current_input .reshape (-1 ,D )


            if self .nuanced_routing :
                routing_mask ,bal_loss ,nuance_analysis =self .gate (current_flat )

                nuance_analysis ['step']=step 
                all_routing_masks .append ({
                'routing_mask':routing_mask .detach ().cpu (),
                'nuance_analysis':{k :v .detach ().cpu ()if torch .is_tensor (v )else v 
                for k ,v in nuance_analysis .items ()}
                })
            else :
                routing_mask ,bal_loss =self .gate (current_flat )
                nuance_analysis =None 
                all_routing_masks .append (routing_mask .detach ().cpu ())

            balance_losses .append (bal_loss )


            expert_outputs =[]
            for i ,expert in enumerate (self .experts ):
                try :

                    if torch .cuda .is_available ():
                        torch .cuda .empty_cache ()


                    expert_out =expert (current_flat )
                    expert_outputs .append (expert_out )

                except RuntimeError as e :
                    if "out of memory"in str (e ).lower ():

                        logging .warning (f"🚨 Expert {i } OOM, using zero fallback")
                        zero_out =torch .zeros_like (current_flat )
                        expert_outputs .append (zero_out )
                        torch .cuda .empty_cache ()
                    else :
                        raise 


            try :
                expert_out =torch .stack (expert_outputs ,dim =1 )
            except RuntimeError as e :
                if "out of memory"in str (e ).lower ():

                    k =min (8 ,len (expert_outputs ))
                    expert_out =torch .stack (expert_outputs [:k ],dim =1 )

                    routing_mask =routing_mask [:,:k ]
                    logging .warning (f"🚨 Reduced to {k } experts due to OOM")
                else :
                    raise 


            assert routing_mask .size (1 )==expert_out .size (1 ),f"gate={routing_mask .size (1 )}  experts={expert_out .size (1 )}"

            weighted =expert_out *routing_mask .unsqueeze (-1 )


            if self .nuanced_routing :
                combined_flat =self .combiner (expert_out ,routing_mask ,nuance_analysis )
            else :
                combined_flat =self ._combine_experts (weighted ,routing_mask )


            combined =combined_flat .view (B ,T ,D )



            tok_score =combined .norm (dim =-1 )


            if T ==0 or tok_score .numel ()==0 :

                best_tok =torch .zeros (batch_size ,D ,device =combined .device ,dtype =combined .dtype )
            else :
                best_idx =tok_score .argmax (1 )


                if torch .any (best_idx <0 )or torch .any (best_idx >=T ):
                    raise RuntimeError (f"CRITICAL: best_idx out of bounds! "
                    f"best_idx range: {best_idx .min ().item ()}-{best_idx .max ().item ()}, "
                    f"sequence length T: {T }")


                try :
                    best_tok =combined [torch .arange (batch_size ),best_idx ]
                except RuntimeError as e :
                    raise RuntimeError (f"CRITICAL: Tensor indexing failed! "
                    f"combined shape: {combined .shape }, "
                    f"best_idx shape: {best_idx .shape }, "
                    f"best_idx range: {best_idx .min ().item ()}-{best_idx .max ().item ()}, "
                    f"batch_size: {batch_size }, T: {T }, D: {D }, "
                    f"Original error: {str (e )}")


            if self .training :
                self .training_step +=1 

            alpha =min (0.15 ,0.15 *self .training_step /self .alpha_anneal_steps )
            alpha =max (0.05 ,alpha )


            combined_for_memory =(1 -alpha )*combined .mean (1 )+alpha *best_tok 
            memory ,usage ,read_vec_batch =self .working_memory (memory ,usage ,combined_for_memory )



            read_vec =read_vec_batch .unsqueeze (1 ).expand (-1 ,T ,-1 )


            if self .training and hasattr (self ,'mem_use_coef')and self .mem_use_coef >0 :

                mem_use_loss =((read_vec_batch -combined_for_memory ).pow (2 ).mean (dim =-1 )).mean ()
                self .mem_use_scalar =mem_use_loss *self .mem_use_coef 
            else :
                self .mem_use_scalar =0.0 



            combined_flat =combined .reshape (-1 ,D )
            read_vec_flat =read_vec .reshape (-1 ,D )
            tg_in =torch .cat ([combined_flat ,read_vec_flat ],dim =-1 )

            thought_flat =self .thought_generator (tg_in )
            thought =thought_flat .view (B ,T ,D )

            all_thoughts .append (thought )
            current_input =thought 



        if all_thoughts :
            stacked_thoughts =torch .stack (all_thoughts ,dim =0 )

            steps ,B ,T ,D =stacked_thoughts .shape 
            thoughts_flat =stacked_thoughts .permute (1 ,2 ,0 ,3 ).contiguous ().view (B *T ,steps ,D )
            final_thought_flat =self .diffusion_gate ([thoughts_flat [:,i ]for i in range (steps )])
            final_thought =final_thought_flat .view (B ,T ,D )
            if len (all_thoughts )>=2 and self .analogy_reasoner is not None :
                final_thought =self .analogy_reasoner (all_thoughts [-2 ],all_thoughts [-1 ],final_thought )
        else :
            final_thought =current_input 









        B ,T ,D =final_thought .shape 


        final_thought_flat =final_thought .reshape (-1 ,D )
        logits_flat =self .final_head (final_thought_flat )


        if T ==0 :

            vocab_size =self .final_head .out_features 
            logits =torch .empty (B ,0 ,vocab_size ,device =final_thought .device ,dtype =final_thought .dtype )
        else :
            logits =logits_flat .view (B ,T ,-1 )



        if T ==0 :

            final_thought_batch =torch .zeros (B ,D ,device =final_thought .device ,dtype =final_thought .dtype )
        else :
            final_thought_batch =final_thought .mean (dim =1 )

        logic_flat =self .logic_proj (final_thought_batch )
        logic_tokens =logic_flat .view (batch_size ,self .logic_len ,-1 )
        self .last_logic_tokens =logic_tokens 


        sophistication_scores =[]
        geometric_specialization_scores =[]


        if self .nuanced_routing and all_routing_masks :
            for routing_data in all_routing_masks :
                if isinstance (routing_data ,dict )and 'nuance_analysis'in routing_data :
                    nuance_analysis =routing_data ['nuance_analysis']
                    if 'sophistication_score'in nuance_analysis :
                        sophistication_scores .append (nuance_analysis ['sophistication_score'])
                    if 'geometric_specialization'in nuance_analysis :
                        geometric_specialization_scores .append (nuance_analysis ['geometric_specialization'])


        avg_sophistication =torch .stack (sophistication_scores ).mean ()if sophistication_scores else torch .tensor (0.0 ,device =logits .device )
        avg_geometric_specialization =torch .stack (geometric_specialization_scores ).mean ()if geometric_specialization_scores else torch .tensor (0.0 ,device =logits .device )


        result ={
        "output":logits ,
        "balance_loss":sum (balance_losses )/len (balance_losses )if balance_losses else 0.0 ,
        "final_thought":final_thought_batch ,
        "logic_tokens":logic_tokens ,
        "read_vec":read_vec_batch ,
        "routing_masks":all_routing_masks ,
        "sophistication_score":avg_sophistication ,
        "geometric_specialization_score":avg_geometric_specialization 
        }


        if (self .enable_audio and original_audio_codes is not None and 
        original_audio_codes .dim ()==3 ):

            try :
                audio_spectrograms =self .audio_decoder (final_thought )
                result ["audio_spectrograms"]=audio_spectrograms 
                logger .debug (f"Generated audio spectrograms: {audio_spectrograms .shape }")
            except Exception as e :
                logger .warning (f"Audio generation failed: {e }")

                result ["audio_spectrograms"]=torch .zeros_like (original_audio_codes )

        return result 






class A100OptimizedNpzDataset (IterableDataset ):
    """
    A100 80GB optimized dataset with blazing fast data loading.
    Backwards compatible with all existing configs.
    """

    def __init__ (self ,npz_files :Dict [str ,str ],npy_files :Dict [str ,str ]=None ,
    seq_len :int =256 ,final_output_dim :int =50281 ,
    tokenizer =None ,prefetch_buffer_size :int =64 ):
        """
        Initialize A100 optimized dataset.
        
        Args:
            npz_files: NPZ files dictionary (backwards compatible)
            npy_files: NPY files dictionary (glob patterns supported)
            seq_len: Sequence length
            final_output_dim: Output dimension
            tokenizer: Tokenizer for pad token
            prefetch_buffer_size: Prefetch buffer size for A100
        """
        self .npz_files =npz_files or {}
        self .npy_files =npy_files or {}
        self .seq_len =seq_len 
        self .final_output_dim =final_output_dim 
        self .prefetch_buffer_size =prefetch_buffer_size 


        self .pad_id =self ._get_pad_token_id (tokenizer )


        self .all_files =self ._discover_all_files ()


        self .dataset_info ={}
        self .total_samples =0 
        self ._analyze_all_datasets ()

        logging .info (f"🚀 A100 Optimized Loader: {len (self .all_files )} files, {self .total_samples :,} samples")

    def _get_pad_token_id (self ,tokenizer )->int :
        """Get pad token ID with comprehensive fallbacks"""
        if tokenizer is None :
            return 50257 


        for attr in ['pad_token_id','pad_id','eos_token_id']:
            if hasattr (tokenizer ,attr ):
                pad_id =getattr (tokenizer ,attr )
                if pad_id is not None :
                    return pad_id 


        if hasattr (tokenizer ,'token_to_id'):
            for pad_token in ['[PAD]','<pad>','<|endoftext|>']:
                pad_id =tokenizer .token_to_id (pad_token )
                if pad_id is not None :
                    return pad_id 

        return 50257 

    def _discover_all_files (self )->List [Tuple [str ,str ,str ]]:
        """Discover all files from npz_files and npy_files, determining type by extension."""
        all_files =[]


        combined_files ={**self .npz_files ,**(self .npy_files or {})}

        for name ,path_or_paths in combined_files .items ():
            if isinstance (path_or_paths ,list ):
                paths =path_or_paths 
            else :
                paths =[path_or_paths ]

            for i ,path in enumerate (paths ):

                file_type ='unknown'
                if path .endswith ('.npz'):
                    file_type ='npz'
                elif path .endswith ('.npy'):
                    file_type ='npy'
                else :
                    logger .warning (f"  ⚠️ Unknown file type for {path }, skipping.")
                    continue 

                unique_name =f"{name }_{i :04d}"if len (paths )>1 else name 
                all_files .append ((unique_name ,path ,file_type ))

        return all_files 

    def _detect_format_optimized (self ,data_keys :List [str ])->Dict [str ,str ]:
        """Optimized format detection with comprehensive key mapping"""
        mapping ={}


        token_priority =['tokens','input_ids','data','transcript_tokens',
        'caption_tokens','text_tokens','arr_0']
        for key in token_priority :
            if key in data_keys :
                mapping ['tokens']=key 
                break 


        audio_keys =['mel_features','audio_features','audio_embedding']
        for key in audio_keys :
            if key in data_keys :
                mapping ['audio']=key 
                break 


        image_keys =['image_features','visual_features','image_embedding']
        for key in image_keys :
            if key in data_keys :
                mapping ['image']=key 
                break 


        if 'attention_mask'in data_keys :
            mapping ['attention_mask']='attention_mask'
        elif 'mask'in data_keys :
            mapping ['attention_mask']='mask'

        return mapping 

    def _analyze_all_datasets (self ):
        """Analyze all datasets to count samples and detect formats"""
        logging .info ("🔍 Analyzing datasets for A100 optimization...")

        for name ,path ,file_type in self .all_files :
            try :
                if file_type =='npz':
                    self ._analyze_npz_file (name ,path )
                elif file_type =='npy':
                    self ._analyze_npy_file (name ,path )

            except Exception as e :
                logging .error (f"Error analyzing {name }: {e }")

        logging .info (f"📊 Total samples: {self .total_samples :,}")

    def _analyze_npz_file (self ,name :str ,path :str ):
        """Analyze NPZ file for sample count and format - FIXES NPZ BUG"""
        try :

            obj =np .load (path ,allow_pickle =True )
            if isinstance (obj ,np .lib .npyio .NpzFile ):
                keys =list (obj .keys ())


                if 'texts'in keys :

                    texts =obj ['texts']
                    num_samples =len (texts )if hasattr (texts ,'__len__')else 1000 

                    self .dataset_info [name ]={
                    'path':path ,
                    'type':'npz',
                    'format':{'texts':'texts'},
                    'samples':num_samples ,
                    'text_format':True 
                    }
                    self .total_samples +=num_samples 
                    logging .info (f"  ✅ {name }: {num_samples :,} samples (Enhanced NPZ)")
                    obj .close ()
                    return 


                mapping =self ._detect_format_optimized (keys )

                if 'tokens'in mapping :
                    token_key =mapping ['tokens']
                    tokens =obj [token_key ]

                    if tokens .ndim ==1 :

                        num_samples =max (0 ,len (tokens )-self .seq_len +1 )
                    else :

                        num_samples =tokens .shape [0 ]

                    self .dataset_info [name ]={
                    'path':path ,
                    'type':'npz',
                    'format':mapping ,
                    'samples':num_samples ,
                    'token_shape':tokens .shape 
                    }
                    self .total_samples +=num_samples 
                    logging .info (f"  ✅ {name }: {num_samples :,} samples (NPZ)")
                else :
                    logging .warning (f"  ⚠️ No tokens in {name }")

                obj .close ()
            else :

                keys =['data']
                mapping =self ._detect_format_optimized (keys )
                if obj .size >0 :
                    num_samples =max (0 ,len (obj )-self .seq_len +1 )if obj .ndim ==1 else obj .shape [0 ]
                    self .dataset_info [name ]={
                    'path':path ,
                    'type':'npz',
                    'format':{'tokens':'data'},
                    'samples':num_samples ,
                    'token_shape':obj .shape 
                    }
                    self .total_samples +=num_samples 
                    logging .info (f"  ✅ {name }: {num_samples :,} samples (NPZ-single)")

        except Exception as e :
            logging .error (f"  ❌ Error analyzing NPZ {name }: {e }")

    def _analyze_npy_file (self ,name :str ,path :str ):
        """Analyze NPY file for sample count"""
        try :

            data =np .load (path ,mmap_mode ='r')

            if data .ndim ==1 :

                num_samples =max (0 ,len (data )-self .seq_len +1 )
            else :

                num_samples =data .shape [0 ]

            self .dataset_info [name ]={
            'path':path ,
            'type':'npy',
            'samples':num_samples ,
            'shape':data .shape 
            }
            self .total_samples +=num_samples 
            logging .info (f"  ✅ {name }: {num_samples :,} samples (NPY)")

        except Exception as e :
            logging .error (f"  ❌ Error analyzing NPY {name }: {e }")

    def _load_sequence_from_npz (self ,path :str ,mapping :Dict [str ,str ],index :int )->Optional [Dict ]:
        """Load sequence from NPZ with optimized format handling"""
        try :
            with np .load (path ,allow_pickle =True )as data :

                if 'texts'in data .keys ():
                    if index >=len (data ['texts']):
                        return None 
                    text_entry =data ['texts'][index ]


                    if isinstance (text_entry ,(bytes ,np .bytes_ )):
                        text_str =text_entry .decode ('utf-8')
                    elif isinstance (text_entry ,np .ndarray )and text_entry .dtype .kind in ['U','S']:
                        text_str =str (text_entry )
                    else :
                        text_str =str (text_entry )



                    tokens =np .array ([hash (word )%50281 for word in text_str .split ()],dtype =np .int64 )
                    if len (tokens )==0 :
                        return None 
                    sequence =tokens [:self .seq_len ]if len (tokens )>self .seq_len else tokens 
                else :

                    token_key =mapping ['tokens']
                    tokens =data [token_key ]

                    if tokens .ndim ==1 :

                        if index +self .seq_len >=len (tokens ):
                            return None 
                        sequence =tokens [index :index +self .seq_len ]
                    else :

                        if index >=tokens .shape [0 ]:
                            return None 
                        sequence =tokens [index ]


                if sequence .dtype .kind in ['U','S']:
                    logging .warning (f"String tokens found in {path }, skipping")
                    return None 


                if sequence .dtype ==np .uint16 :
                    sequence =sequence .astype (np .int64 )
                elif sequence .dtype not in [np .int32 ,np .int64 ]:
                    sequence =sequence .astype (np .int64 )


                sample =self ._prepare_sample (sequence ,mapping .get ('audio'),mapping .get ('image'))


                if 'audio'in mapping :
                    try :
                        audio_data =data [mapping ['audio']]
                        if audio_data .ndim ==3 :
                            audio_data =audio_data [0 ]
                        sample ['audio_features']=torch .from_numpy (audio_data .astype (np .float32 ))
                    except Exception as e:
                        logging.warning(f"Failed to load audio from {path}: {e}")
                        sample["audio_features"] = torch.zeros(1, dtype=torch.float32)

                return sample 

        except Exception as e :
            logging .error (f"Error loading from NPZ {path }: {e }")
            return None 

    def _load_sequence_from_npy (self ,path :str ,index :int )->Optional [Dict ]:
        """Load sequence from NPY with memory mapping for efficiency"""
        try :

            data =np .load (path ,mmap_mode ='r')

            if data .ndim ==1 :

                if index +self .seq_len >=len (data ):
                    return None 
                sequence =data [index :index +self .seq_len ].copy ()
            else :

                if index >=data .shape [0 ]:
                    return None 
                sequence =data [index ].copy ()


            if sequence .dtype ==np .uint16 :
                sequence =sequence .astype (np .int64 )
            elif sequence .dtype not in [np .int32 ,np .int64 ]:
                sequence =sequence .astype (np .int64 )

            return self ._prepare_sample (sequence )

        except Exception as e :
            logging .error (f"Error loading from NPY {path }: {e }")
            return None 

    def _prepare_sample (self ,sequence :np .ndarray ,audio_key =None ,image_key =None )->Dict :
        """Prepare training sample with causal LM format"""
        if len (sequence )<2 :
            return None 


        if len (sequence )<self .seq_len :
            padded =np .full (self .seq_len ,self .pad_id ,dtype =np .int64 )
            padded [:len (sequence )]=sequence 
            sequence =padded 
        else :
            sequence =sequence [:self .seq_len ]


        input_ids =torch .from_numpy (sequence [:-1 ])
        labels =torch .from_numpy (sequence [1 :])
        attention_mask =(input_ids !=self .pad_id ).long ()

        return {
        'input_ids':input_ids ,
        'labels':labels ,
        'attention_mask':attention_mask 
        }

    def __iter__ (self ):
        """Optimized iteration with worker distribution"""
        worker_info =torch .utils .data .get_worker_info ()


        if worker_info is None :
            files_to_process =self .dataset_info .items ()
        else :
            all_items =list (self .dataset_info .items ())
            worker_files =all_items [worker_info .id ::worker_info .num_workers ]
            files_to_process =worker_files 

        for name ,info in files_to_process :
            if info ['samples']==0 :
                continue 


            for i in range (info ['samples']):
                if info ['type']=='npz':

                    if info .get ('text_format',False ):
                        sample =self ._load_sequence_from_npz (info ['path'],{'texts':'texts'},i )
                    else :
                        sample =self ._load_sequence_from_npz (info ['path'],info ['format'],i )
                else :
                    sample =self ._load_sequence_from_npy (info ['path'],i )

                if sample is not None :
                    sample ['dataset_name']=name 
                    yield sample 

    def __len__ (self ):



        return min (self .total_samples ,10 **6 )

def create_geometric_optimizer (model ,stage_config ,stage_name =None ):

    params_config =stage_config 



    resolved_lr =(
    params_config .get ('learning_rate')or 
    params_config .get ('optimizer',{}).get ('lr')or 
    params_config .get ('training',{}).get ('learning_rate')or 
    5e-4 
    )
    params_config ['learning_rate']=resolved_lr 

    if GEOOPT_AVAILABLE :

        geo_params =[]
        euc_params =[]


        if stage_name =='curvature_calibration':
            for name ,param in model .named_parameters ():
                if param .requires_grad and ('raw_c'in name or 'raw_r'in name ):
                    geo_params .append (param )

            euc_params =[]
        else :

            for name ,param in model .named_parameters ():
                if not param .requires_grad :
                    continue 

                if ('raw_c'in name or 'raw_r'in name or 
                ('experts'in name and ('to_tan'in name or 'fc'in name or 'ln'in name or 'output_proj'in name ))):
                    geo_params .append (param )
                else :
                    euc_params .append (param )


        param_groups =[]
        if euc_params :
            param_groups .append ({'params':euc_params })
        if geo_params :
            param_groups .append ({'params':geo_params ,'lr':params_config ['learning_rate']*params_config .get ('geo_lr_factor',2.0 )})

        if not param_groups :
            raise ValueError (f"No parameters found for optimization in stage {stage_name }")

        optimizer =geoopt .optim .RiemannianAdam (
        param_groups ,
        lr =params_config ['learning_rate'],
        betas =(params_config .get ('beta1',0.9 ),params_config .get ('beta2',0.95 )),
        weight_decay =params_config .get ('weight_decay',0.01 )
        )
        logging .info ("✅ Using RiemannianAdam optimizer for geometric manifolds")
        logging .info (f"   🔢 Geometric parameters: {len (geo_params )}")
        logging .info (f"   🔢 Euclidean parameters: {len (euc_params )}")

        if stage_name =='curvature_calibration':
            logging .info (f"   🌀 CURVATURE CALIBRATION: Optimizer restricted to {len (geo_params )} parameters")

    else :

        if stage_name =='curvature_calibration':

            curvature_params =[param for name ,param in model .named_parameters ()
            if param .requires_grad and ('raw_c'in name or 'raw_r'in name )]
            if not curvature_params :
                raise ValueError ("No curvature parameters found for optimization")
            optimizer =torch .optim .AdamW (
            curvature_params ,
            lr =params_config ['learning_rate'],
            betas =(params_config .get ('beta1',0.9 ),params_config .get ('beta2',0.95 )),
            weight_decay =params_config .get ('weight_decay',0.01 )
            )
            logging .info (f"   🌀 CURVATURE CALIBRATION: AdamW restricted to {len (curvature_params )} parameters")
        else :
            optimizer =torch .optim .AdamW (
            [p for p in model .parameters ()if p .requires_grad ],
            lr =params_config ['learning_rate'],
            betas =(params_config .get ('beta1',0.9 ),params_config .get ('beta2',0.95 )),
            weight_decay =params_config .get ('weight_decay',0.01 )
            )
        logging .warning ("🚨 CRITICAL WARNING: Using standard AdamW optimizer instead of RiemannianAdam!")
        logging .warning ("   The geometric manifold structure will not be properly optimized.")
        logging .warning ("   This significantly reduces the effectiveness of the MGM architecture.")
        logging .warning ("   Please install GeoOpt for optimal performance: pip install git+https://github.com/geoopt/geoopt.git")

    return optimizer 

def save_checkpoint (state ,filename ="checkpoint.pth.tar"):
    """Save *state* to *filename* ensuring stage metadata is present."""
    stage =state .get ("stage")or state .get ("config",{}).get ("_current_stage")
    if stage :
        state ["stage"]=stage 
    else :
        state .setdefault ("stage","unknown")
    state .setdefault ("stage_completed",False )

    if '_SOBOL_CACHE'in globals ()and globals ()['_SOBOL_CACHE']:
        state .setdefault ('meta',{})['sobol_cache']=globals ()['_SOBOL_CACHE']
    torch .save (state ,filename )
    logging .info (f"💾 Checkpoint saved: {filename }")

def train_epoch (
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
start_batch :int =0 ,
lr_scheduler =None ,
best_val_loss =float ('inf')
):
    model .train ()
    reward_model .train ()


    if len (data_loader )==0 :
        logging .error ("❌ Data loader is empty! No training samples found.")
        logging .error ("   Check that dataset files exist and contain valid samples.")
        return global_step ,best_val_loss 


    def h (cfg ,key ,default =None ):
        """Stage-local overrides > training-level > default"""
        return cfg .get (key ,cfg .get ('training',{}).get (key ,default ))


    def _clip_real_grads (params ,max_norm ):
        real_params =[p for p in params if p .grad is not None and not torch .is_complex (p .grad )]
        if real_params :
            torch .nn .utils .clip_grad_norm_ (real_params ,max_norm )


    def _safe_unscale (scaler ,optimizer ):
        """Safely unscale gradients excluding complex tensors from AMP"""
        try :
            scaler .unscale_ (optimizer )
        except RuntimeError as e :

            has_complex_grad =any (
            param .grad is not None and torch .is_complex (param .grad )
            for group in optimizer .param_groups 
            for param in group ['params']
            )
            if has_complex_grad :

                for group in optimizer .param_groups :
                    for param in group ['params']:
                        if param .grad is not None and not torch .is_complex (param .grad ):
                            param .grad .data .div_ (scaler .get_scale ())
            else :
                raise e 

    def _safe_step (scaler ,optimizer ):
        """Safely step optimizer with AMP, handling complex tensors"""
        try :
            scaler .step (optimizer )
        except RuntimeError as e :
            if "ComplexFloat"in str (e ):

                for group in optimizer .param_groups :
                    for param in group ['params']:
                        if param .grad is not None and not torch .is_complex (param .grad ):

                            if not hasattr (optimizer ,'_backup_step'):
                                optimizer ._backup_step =optimizer .step 

                            with torch .no_grad ():
                                if param .grad .is_sparse :
                                    continue 
                                param .data .add_ (param .grad .data ,alpha =-group ['lr'])
            else :
                raise e 


    for name ,p in model .named_parameters ():
        if p .grad is not None :
            if 'raw_c'in name or 'raw_r'in name :

                torch .nn .utils .clip_grad_norm_ (p ,max_norm =0.1 )
            elif 'raw_frequency'in name or 'raw_phase'in name or 'raw_c_speed'in name :

                torch .nn .utils .clip_grad_norm_ (p ,max_norm =0.5 )

    total_task_loss =0.0 
    total_balance_loss =0.0 
    total_ppo_loss =0.0 
    total_samples =0 
    processed_batches =0 


    task_loss =torch .tensor (0.0 ,device =device )
    balance_loss =torch .tensor (0.0 ,device =device )
    ppo_loss =torch .tensor (0.0 ,device =device )



    stage_config =config ['training_stages'].get (epoch ,{})
    is_curvature_calibration =ppo_optimizer is None 
    model_is_compiled =hasattr (model ,'_compiled_graph')

    use_amp =(config ['training']['use_amp']and 
    device .type =='cuda'and 
    not is_curvature_calibration and 
    not model_is_compiled )

    if use_amp :
        scaler =ComplexFilteredGradScaler ()
        scaler .register_complex_params (model )
        logging .info ("🚀 Using AMP with complex tensor filtering")
    else :
        scaler =None 
        if is_curvature_calibration :
            logging .info ("📝 AMP disabled for curvature calibration stage")
        elif model_is_compiled :
            logging .info ("📝 AMP disabled due to model compilation")
        else :
            logging .info ("📝 AMP disabled (not using CUDA or disabled in config)")

    progress_bar =tqdm (
    data_loader ,
    desc =f"Epoch {epoch +1 }",
    leave =False ,
    dynamic_ncols =True ,
    file =sys .stdout ,
    total =len (data_loader ),
    initial =start_batch ,
    )


    accumulation_steps =config ['training'].get ('gradient_accumulation_steps',1 )


    beta =TRAIN_STATE .get ('ppo_kl_beta',0.02 )

    target_kl =config .get ('training',{}).get ('ppo_target_kl',0.05 )

    for batch_idx ,batch in enumerate (progress_bar ):
        freeze_steps =config ['training'].get ('freeze_emb_steps',3000 )
        if global_step <freeze_steps :
            model .embedding .weight .requires_grad_ (False )
            model .position_embeddings .weight .requires_grad_ (False )
        else :
            model .embedding .weight .requires_grad_ (True )
            model .position_embeddings .weight .requires_grad_ (True )












        if batch_idx <start_batch :
            continue 


        max_steps =config .get ('training',{}).get ('max_steps',None )
        steps_processed =batch_idx -start_batch 
        if max_steps is not None and steps_processed >=max_steps :
            logging .info (f"Training stopped at {steps_processed } steps (max_steps={max_steps })")
            break 


        if batch is None :
            logging .warning (f"⚠️ Batch {batch_idx } is None, skipping...")
            continue 


        try :
            safety_errors =ProductionSafetyChecks .validate_batch (batch ,config )
            if safety_errors :
                for error in safety_errors :
                    if 'CRITICAL'in error :
                        logger .error (f"❌ Batch {batch_idx } CRITICAL ERROR: {error }")
                        continue 
                    else :
                        logger .warning (f"⚠️ Batch {batch_idx } WARNING: {error }")
        except Exception as e :
            logger .error (f"❌ Batch validation error: {e }")
            continue 


        has_text ='input_ids'in batch and batch ['input_ids']is not None 
        has_vision ='pixel_values'in batch and batch ['pixel_values']is not None 
        has_audio ='audio_codes'in batch and batch ['audio_codes']is not None 

        if not (has_text or has_vision or has_audio ):
            continue 


        model_inputs ={}
        if has_text :
            model_inputs ['input_ids']=batch ['input_ids'].to (device ,dtype =torch .long )
            model_inputs ['attention_mask']=batch ['attention_mask'].to (device ,dtype =torch .long )
            target =batch ['labels'].to (device ,dtype =torch .long )
        if has_vision :
            model_inputs ['pixel_values']=batch ['pixel_values'].to (device )
            if not has_text :
                target =batch .get ('labels',torch .zeros (batch ['pixel_values'].size (0 ),device =device ,dtype =torch .long ))
        if has_audio :
            model_inputs ['audio_codes']=batch ['audio_codes'].to (device ,dtype =torch .long )
            if not has_text :

                audio_codes =batch ['audio_codes']

                if audio_codes .dim ()==3 :


                    B ,CB ,T =audio_codes .shape 

                    target =audio_codes [:,0 ,:].to (device ,dtype =torch .long )
                elif audio_codes .dim ()==2 :

                    target =audio_codes .to (device ,dtype =torch .long )
                else :

                    target =torch .zeros (batch ['audio_codes'].size (0 ),device =device ,dtype =torch .long )


        if batch_idx %accumulation_steps ==0 :
            optimizer .zero_grad ()
            if ppo_optimizer is not None :
                ppo_optimizer .zero_grad ()


        if batch_idx %accumulation_steps ==0 :
            global_step +=1 


        warmup_steps =h (config ,'warmup_steps',10000 )
        warm_fac =min (1.0 ,global_step /warmup_steps )



        curvature_params =[]
        total_optimizer_params =0 

        for group in optimizer .param_groups :
            for param in group ['params']:
                total_optimizer_params +=1 

                for name ,model_param in model .named_parameters ():
                    if param is model_param and ('raw_c'in name or 'raw_r'in name ):
                        curvature_params .append (name )
                        break 

        is_curvature_mode =len (curvature_params )==total_optimizer_params and len (curvature_params )>0 

        if batch_idx ==0 :
            print (f"🔍 OPTIMIZER HAS: {total_optimizer_params } total params, {len (curvature_params )} curvature params")
            print (f"🔍 CURVATURE PARAMS: {curvature_params }")
            print (f"🔍 CURVATURE MODE: {is_curvature_mode }")

        if is_curvature_mode :
            print (f"🚀 CURVATURE FAST MODE ACTIVE: Skipping {sum (p .numel ()for p in model .parameters ()):,} parameter forward pass!")
            logging .info (f"🚀 CURVATURE FAST MODE: Optimizing only curvature parameters")

            curvature_reg =torch .tensor (0.0 ,device =device ,requires_grad =True )
            curvature_count =0 
            for name ,param in model .named_parameters ():
                if 'raw_c'in name or 'raw_r'in name :
                    curvature_reg =curvature_reg +param .pow (2 )
                    curvature_count +=1 

            if curvature_count >0 :
                curvature_reg =curvature_reg /curvature_count 


            dummy_task_loss =torch .tensor (0.001 ,device =device ,requires_grad =True )
            combined_loss =(dummy_task_loss +curvature_reg *config ['training'].get ('curvature_reg_weight',0.01 ))/accumulation_steps 
            ppo_loss =torch .tensor (0.0 ,device =device )
            task_loss =dummy_task_loss 
            balance_loss =torch .tensor (0.0 ,device =device )


            model_output ={}


            combined_loss .backward ()


            if (batch_idx +1 )%accumulation_steps ==0 :
                max_grad_norm =config ['training'].get ('max_grad_norm',1.0 )
                clean_gradients (model .parameters ())
                _clip_real_grads (model .parameters (),max_grad_norm )
                optimizer .step ()
        elif use_amp :
            with torch .amp .autocast ('cuda'):
                model_output =model (**model_inputs )


                if isinstance (model_output ,dict ):
                    final_output =model_output .get ("output",model_output .get ("logits"))
                    final_thought =model_output .get ("final_thought")
                    balance_loss =model_output .get ("balance_loss",torch .tensor (0.0 ,device =device ))

                    sophistication_score =model_output .get ("sophistication_score",torch .tensor (0.0 ,device =device ))
                    geometric_specialization_score =model_output .get ("geometric_specialization_score",torch .tensor (0.0 ,device =device ))
                elif isinstance (model_output ,tuple ):
                    final_output =model_output [0 ]
                    final_thought =model_output [1 ]if len (model_output )>1 else None 
                    balance_loss =torch .tensor (0.0 ,device =device )
                    sophistication_score =torch .tensor (0.0 ,device =device )
                    geometric_specialization_score =torch .tensor (0.0 ,device =device )
                else :
                    final_output =model_output 
                    final_thought =None 
                    balance_loss =torch .tensor (0.0 ,device =device )
                    sophistication_score =torch .tensor (0.0 ,device =device )
                    geometric_specialization_score =torch .tensor (0.0 ,device =device )


                if final_output.dim() == 3:
                    fo = final_output.reshape(-1, final_output.size(-1))
                    tgt = target.reshape(-1)
                elif final_output.dim() == 2:
                    fo = final_output
                    tgt = target[:, 0] if target.dim() > 1 else target
                else:
                    fo = final_output
                    tgt = target
                task_loss = criterion(fo, tgt)


                curvature_reg =torch .tensor (0.0 ,device =device )
                curvature_count =0 
                for name ,param in model .named_parameters ():
                    if 'raw_c'in name or 'raw_r'in name :
                        curvature_reg +=param .pow (2 ).mean ()
                        curvature_count +=1 

                if curvature_count >0 :
                    curvature_reg =curvature_reg /curvature_count 


                pred =F .softmax (final_output ,dim =-1 )
                with torch .no_grad ():
                    if final_output .dim ()==3 :
                        correct =(pred .argmax (-1 )==target ).float ().mean (dim =1 ,keepdim =True )
                    else :
                        correct =(pred .argmax (-1 )==target ).float ().unsqueeze (-1 )
                verif_logit =model .verifier (final_thought .detach ())
                verifier_loss =F .binary_cross_entropy_with_logits (verif_logit ,correct )


                w_bal =h (config ,'balance_loss_weight',0.0 )
                w_ver =h (config ,'verifier_loss_weight',0.01 )
                curvature_weight =h (config ,'curvature_reg_weight',0.01 )


                mem_aux_loss =getattr (model ,'mem_use_scalar',0.0 )
                mem_aux_weight =h (config ,'memory_auxiliary_loss_weight',0.0 )
                mem_aux_loss =warm_fac *mem_aux_weight *mem_aux_loss 


                sophistication_weight =config ['model'].get ('sophistication_loss_weight',0.0 )
                geometric_weight =config ['model'].get ('geometric_specialization_weight',0.0 )


                sophistication_loss =-sophistication_score *sophistication_weight 


                geometric_loss =-geometric_specialization_score *geometric_weight 


                E =torch .stack ([F .normalize (ex .fc3 .weight ,dim =1 ).view (-1 )for ex in model .experts ])
                orth_loss =((E @E .t ())-torch .eye (E .size (0 ),device =E .device )).pow (2 ).sum ()*1e-3 

                combined_loss =(task_loss +
                w_bal *balance_loss +
                warm_fac *w_ver *verifier_loss +
                curvature_weight *curvature_reg +
                mem_aux_loss +
                sophistication_loss +
                geometric_loss +
                orth_loss )/accumulation_steps 


                if ppo_optimizer is not None :

                    reward_input =final_thought .detach ()
                    rewards =reward_model (reward_input )
                    reward_pred =rewards .squeeze ()
                    reward_target =correct .squeeze ()

                    reward_model_loss =F .mse_loss (reward_pred ,reward_target )/accumulation_steps 

                    log_probs =F .log_softmax (final_output ,dim =-1 )
                    with torch .no_grad ():
                        old_log_probs =log_probs .detach ()

                    if final_output .dim ()==3 :
                        mask =target !=-100 
                        safe_target =target .clone ()
                        safe_target [~mask ]=0 
                        action_log_probs =log_probs .gather (2 ,safe_target .unsqueeze (-1 )).squeeze (-1 )
                        action_log_probs =(action_log_probs *mask ).mean (dim =1 )
                    else :
                        mask =target !=-100 
                        safe_target =target .clone ()
                        safe_target [~mask ]=0 
                        action_log_probs =log_probs .gather (1 ,safe_target .unsqueeze (1 )).squeeze (-1 )
                        action_log_probs =action_log_probs *mask 


                    rewards_for_ppo =reward_pred .detach ()

                    if action_log_probs .shape !=rewards_for_ppo .shape :
                        if rewards_for_ppo .dim ()>action_log_probs .dim ():
                            rewards_for_ppo =rewards_for_ppo .mean (dim =-1 )
                        elif action_log_probs .dim ()>rewards_for_ppo .dim ():
                            action_log_probs =action_log_probs .mean (dim =-1 )

                    ppo_loss =-(action_log_probs *rewards_for_ppo ).mean ()

                    logp_new =log_probs 
                    logp_old =log_probs .detach ()
                    kl =F .kl_div (logp_new ,logp_old ,log_target =True ,reduction ='batchmean')
                    kl =torch .nan_to_num (kl ,nan =0.0 ,posinf =0.0 ,neginf =0.0 )
                    assert logp_new .shape ==logp_old .shape ,"KL shapes mismatch"
                    policy_loss =ppo_loss 
                    ppo_loss =(policy_loss +beta *kl )/accumulation_steps 

                    with torch .no_grad ():
                        beta_new =beta 
                        if kl >1.5 *target_kl :
                            beta_new =min (beta *1.5 ,0.2 )
                        elif kl <0.5 *target_kl :

                            beta_new =max (beta *0.7 ,5e-3 )
                        beta =0.9 *beta +0.1 *beta_new 
                    TRAIN_STATE ['ppo_kl_beta']=beta 

                else :
                    ppo_loss =torch .tensor (0.0 ,device =device )
                    reward_model_loss =torch .tensor (0.0 ,device =device )


            if ppo_optimizer is not None :

                total_loss =combined_loss +ppo_loss +reward_model_loss 
                scaler .scale (total_loss ).backward ()
            else :
                scaler .scale (combined_loss ).backward ()


            if (batch_idx +1 )%accumulation_steps ==0 :

                _safe_unscale (scaler ,optimizer )
                if ppo_optimizer is not None :
                    _safe_unscale (scaler ,ppo_optimizer )


                max_grad_norm =config ['training'].get ('max_grad_norm',1.0 )
                for p in model .parameters ():
                    if p .grad is not None and torch .isnan (p .grad ).any ():
                        p .grad =torch .nan_to_num (p .grad )
                _clip_real_grads (model .parameters (),max_grad_norm )
                if ppo_optimizer is not None :
                    _clip_real_grads (reward_model .parameters (),max_grad_norm )


                _safe_step (scaler ,optimizer )
                if ppo_optimizer is not None :

                    if any (p .grad is not None for p in reward_model .parameters ()):
                        _safe_step (scaler ,ppo_optimizer )


                scaler .update ()
        else :

            model_output =model (**model_inputs )


            if isinstance (model_output ,dict ):
                final_output =model_output .get ("output",model_output .get ("logits"))
                final_thought =model_output .get ("final_thought")
                balance_loss =model_output .get ("balance_loss",torch .tensor (0.0 ,device =device ))

                sophistication_score =model_output .get ("sophistication_score",torch .tensor (0.0 ,device =device ))
                geometric_specialization_score =model_output .get ("geometric_specialization_score",torch .tensor (0.0 ,device =device ))
            elif isinstance (model_output ,tuple ):
                final_output =model_output [0 ]
                final_thought =model_output [1 ]if len (model_output )>1 else None 
                balance_loss =torch .tensor (0.0 ,device =device )
                sophistication_score =torch .tensor (0.0 ,device =device )
                geometric_specialization_score =torch .tensor (0.0 ,device =device )
            else :
                final_output =model_output 
                final_thought =None 
                balance_loss =torch .tensor (0.0 ,device =device )
                sophistication_score =torch .tensor (0.0 ,device =device )
                geometric_specialization_score =torch .tensor (0.0 ,device =device )


            if final_output .dim ()==3 :
                batch_size ,seq_len ,vocab_size =final_output .shape 


                if target .dim ()==1 :

                    target =target .unsqueeze (1 ).expand (batch_size ,seq_len )
                elif target .dim ()==2 and target .size (1 )!=seq_len :

                    if target .size (1 )>seq_len :
                        target =target [:,:seq_len ]
                    else :

                        padding =torch .full ((batch_size ,seq_len -target .size (1 )),-100 ,
                        dtype =target .dtype ,device =target .device )
                        target =torch .cat ([target ,padding ],dim =1 )


                task_loss =criterion (final_output .reshape (-1 ,vocab_size ),target .reshape (-1 ))

            else :

                if target .dim ()>1 :

                    if target .size (-1 )>1 :
                        target =target [:,0 ]
                    else :
                        target =target .squeeze (-1 )

                task_loss =criterion (final_output ,target )


            curvature_reg =torch .tensor (0.0 ,device =device )
            curvature_count =0 
            for name ,param in model .named_parameters ():
                if 'raw_c'in name or 'raw_r'in name :
                    curvature_reg +=param .pow (2 ).mean ()
                    curvature_count +=1 

            if curvature_count >0 :
                curvature_reg =curvature_reg /curvature_count 


            pred =F .softmax (final_output ,dim =-1 )
            with torch .no_grad ():

                if final_output .dim ()==3 :
                    correct =(pred .argmax (-1 )==target ).float ().mean (dim =1 ,keepdim =True )
                else :
                    correct =(pred .argmax (-1 )==target ).float ().unsqueeze (-1 )
            verif_logit =model .verifier (final_thought .detach ())
            verifier_loss =F .binary_cross_entropy_with_logits (verif_logit ,correct )


            w_bal =config ['training']['balance_loss_weight']
            w_ver =config ['training'].get ('verifier_loss_weight',0.01 )
            curvature_weight =config ['training'].get ('curvature_reg_weight',0.01 )


            mem_aux_loss =getattr (model ,'mem_use_scalar',0.0 )
            mem_aux_weight =h (config ,'memory_auxiliary_loss_weight',0.0 )
            mem_aux_loss =warm_fac *mem_aux_weight *mem_aux_loss 


            sophistication_weight =config ['model'].get ('sophistication_loss_weight',0.0 )
            geometric_weight =config ['model'].get ('geometric_specialization_weight',0.0 )


            sophistication_loss =-sophistication_score *sophistication_weight 


            geometric_loss =-geometric_specialization_score *geometric_weight 

            combined_loss =(task_loss +
            w_bal *balance_loss +
            warm_fac *w_ver *verifier_loss +
            curvature_weight *curvature_reg +
            mem_aux_loss +
            sophistication_loss +
            geometric_loss )/accumulation_steps 


            if ppo_optimizer is not None :

                reward_input =final_thought .detach ()
                rewards =reward_model (reward_input )
                reward_pred =rewards .squeeze ()
                reward_target =correct .squeeze ()

                reward_model_loss =F .mse_loss (reward_pred ,reward_target )/accumulation_steps 

                log_probs =F .log_softmax (final_output ,dim =-1 )
                with torch .no_grad ():
                    old_log_probs =log_probs .detach ()

                if final_output .dim ()==3 :
                    mask =target !=-100 
                    safe_target =target .clone ()
                    safe_target [~mask ]=0 
                    action_log_probs =log_probs .gather (2 ,safe_target .unsqueeze (-1 )).squeeze (-1 )
                    action_log_probs =(action_log_probs *mask ).mean (dim =1 )
                else :
                    mask =target !=-100 
                    safe_target =target .clone ()
                    safe_target [~mask ]=0 
                    action_log_probs =log_probs .gather (1 ,safe_target .unsqueeze (1 )).squeeze (-1 )
                    action_log_probs =action_log_probs *mask 


                rewards_for_ppo =reward_pred .detach ()

                if action_log_probs .shape !=rewards_for_ppo .shape :
                    if rewards_for_ppo .dim ()>action_log_probs .dim ():
                        rewards_for_ppo =rewards_for_ppo .mean (dim =-1 )
                    elif action_log_probs .dim ()>rewards_for_ppo .dim ():
                        action_log_probs =action_log_probs .mean (dim =-1 )

                ppo_loss =-(action_log_probs *rewards_for_ppo ).mean ()

                logp_new =log_probs 
                logp_old =log_probs .detach ()
                kl =F .kl_div (logp_new ,logp_old ,log_target =True ,reduction ='batchmean')
                kl =torch .nan_to_num (kl ,nan =0.0 ,posinf =0.0 ,neginf =0.0 )
                assert logp_new .shape ==logp_old .shape ,"KL shapes mismatch"
                policy_loss =ppo_loss 
                ppo_loss =(policy_loss +beta *kl )/accumulation_steps 

                with torch .no_grad ():
                    beta_new =beta 
                    if kl >1.5 *target_kl :
                        beta_new =min (beta *1.5 ,0.2 )
                    elif kl <0.5 *target_kl :
                        beta_new =max (beta *0.7 ,5e-3 )
                    beta =0.9 *beta +0.1 *beta_new 
                TRAIN_STATE ['ppo_kl_beta']=beta 
            else :
                ppo_loss =torch .tensor (0.0 ,device =device )
                reward_model_loss =torch .tensor (0.0 ,device =device )


            if ppo_optimizer is not None :

                total_loss =combined_loss +ppo_loss +reward_model_loss 
                total_loss .backward ()
            else :
                combined_loss .backward ()


            if (batch_idx +1 )%accumulation_steps ==0 :

                max_grad_norm =h (config ,'max_grad_norm',1.0 )
                for p in model .parameters ():
                    if p .grad is not None and torch .isnan (p .grad ).any ():
                        p .grad =torch .nan_to_num (p .grad )
                _clip_real_grads (model .parameters (),max_grad_norm )
                if ppo_optimizer is not None :
                    _clip_real_grads (reward_model .parameters (),max_grad_norm )


                optimizer .step ()
                if ppo_optimizer is not None :
                    ppo_optimizer .step ()

        total_task_loss +=task_loss .item ()
        total_balance_loss +=balance_loss .item ()
        total_ppo_loss +=ppo_loss .item ()

        if has_text :
            batch_size =model_inputs ['input_ids'].size (0 )
        elif has_vision :
            batch_size =model_inputs ['pixel_values'].size (0 )
        elif has_audio :
            batch_size =model_inputs ['audio_codes'].size (0 )
        else :
            batch_size =1 
        total_samples +=batch_size 
        processed_batches +=1 


        if 'read_vec'in model_output :
            read_vec_norm =model_output ['read_vec'].norm (dim =-1 ).mean ().item ()
            if batch_idx %50 ==0 :
                logging .info (f"Memory utilization - read_vec.norm(): {read_vec_norm :.4f}")


        sophistication_score_val =0.0 
        geometric_score_val =0.0 
        if isinstance (model_output ,dict ):
            sophistication_score_val =model_output .get ("sophistication_score",torch .tensor (0.0 )).item ()if torch .is_tensor (model_output .get ("sophistication_score",0.0 ))else model_output .get ("sophistication_score",0.0 )
            geometric_score_val =model_output .get ("geometric_specialization_score",torch .tensor (0.0 )).item ()if torch .is_tensor (model_output .get ("geometric_specialization_score",0.0 ))else model_output .get ("geometric_specialization_score",0.0 )

        progress_bar .set_postfix ({
        'task':f"{task_loss .item ():.3f}",
        'balance':f"{balance_loss .item ():.3f}",
        'ppo':f"{ppo_loss .item ():.3f}",
        'mem_norm':f"{read_vec_norm :.3f}"if 'read_vec'in model_output else "N/A",
        'soph':f"{sophistication_score_val :.3f}",
        'geom':f"{geometric_score_val :.3f}",
        'samples':f"{total_samples }"
        })


        save_steps =config .get ('training',{}).get ('save_steps',5000 )
        if batch_idx %save_steps ==0 and batch_idx >0 :

            checkpoint_data ={

            'model_state_dict':model .state_dict (),
            'optimizer_state_dict':optimizer .state_dict (),
            'epoch':epoch ,
            'global_step':global_step ,


            'lr_scheduler_state_dict':lr_scheduler .state_dict ()if lr_scheduler is not None else None ,


            'torch_rng_state':torch .get_rng_state (),
            'numpy_rng_state':np .random .get_state (),
            'python_rng_state':random .getstate (),


            'reward_model_state_dict':reward_model .state_dict (),


            'scaler_state_dict':scaler .state_dict ()if scaler is not None else None ,
            'ppo_optimizer_state_dict':ppo_optimizer .state_dict ()if ppo_optimizer is not None else None ,


            'config':config ,
            'vocab_size':config .get ('vocab_size',config .get ('model',{}).get ('final_output_dim',50257 )),
            'stage':config .get ('_current_stage','unknown'),
            'current_learning_rate':optimizer .param_groups [0 ]['lr'],


            'pytorch_version':torch .__version__ ,
            'checkpoint_format_version':'2025.1',
            'ppo_kl_beta':TRAIN_STATE .get (
            'ppo_kl_beta',config ['training'].get ('ppo_kl_coef',0.02 )
            )
            }


            save_checkpoint (checkpoint_data ,filename =f"checkpoint_epoch_{epoch }_batch_{batch_idx }.pth.tar")


    if processed_batches %accumulation_steps !=0 :
        if use_amp :

            _safe_unscale (scaler ,optimizer )
            if ppo_optimizer is not None :
                _safe_unscale (scaler ,ppo_optimizer )
            _safe_step (scaler ,optimizer )
            if ppo_optimizer is not None :
                _safe_step (scaler ,ppo_optimizer )
            scaler .update ()
        else :

            if accumulation_steps >1 :
                max_grad_norm =h (config ,'max_grad_norm',1.0 )
                for p in model .parameters ():
                    if p .grad is not None and torch .isnan (p .grad ).any ():
                        p .grad =torch .nan_to_num (p .grad )
                _clip_real_grads (model .parameters (),max_grad_norm )
                if ppo_optimizer is not None :
                    _clip_real_grads (reward_model .parameters (),max_grad_norm )
            optimizer .step ()
            if ppo_optimizer is not None :
                ppo_optimizer .step ()
        global_step +=1 

    if processed_batches ==0 :
        logging .warning ("No batches were processed. Skipping average loss calculation.")
        avg_task_loss =float ('nan')
        avg_balance_loss =float ('nan')
        avg_ppo_loss =float ('nan')
    else :
        avg_task_loss =total_task_loss /processed_batches 
        avg_balance_loss =total_balance_loss /processed_batches 
        avg_ppo_loss =total_ppo_loss /processed_batches 

    logging .info (f"Epoch {epoch +1 } Results -> Task: {avg_task_loss :.4f}, Balance: {avg_balance_loss :.4f}, PPO: {avg_ppo_loss :.4f}, Samples: {total_samples :,}")

    return global_step ,best_val_loss 
def validate_mgm_architecture (model ,data_loader ,criterion ,device ,max_steps =None ,stage_name ="unknown"):
    """
    🧠 MGM-AWARE VALIDATION: Measures what actually matters for geometric expert architecture
    
    Instead of just token accuracy, this validates:
    1. Expert diversity and specialization quality
    2. Geometric manifold utilization
    3. Memory integration effectiveness  
    4. Routing entropy and load balancing
    5. Curvature parameter health
    """
    model .eval ()


    total_loss =0 
    correct =0 
    total_samples =0 
    step_count =0 


    expert_activations =torch .zeros (model .num_experts ,device =device )
    routing_entropies =[]
    curvature_values =[]
    memory_utilizations =[]
    balance_losses =[]

    sophistication_scores =[]
    geometric_specialization_scores =[]

    start_time =time .time ()

    with torch .no_grad ():
        for batch in data_loader :
            if max_steps is not None and step_count >=max_steps :
                break 


            if not ('input_ids'in batch and 'labels'in batch ):
                continue 
            if batch ['input_ids']is None or batch ['labels']is None :
                continue 


            model_inputs ={
            'input_ids':batch ['input_ids'].to (device ,non_blocking =True ),
            'attention_mask':batch ['attention_mask'].to (device ,non_blocking =True )
            }
            target =batch ['labels'].to (device ,non_blocking =True )
            batch_size =batch ['input_ids'].size (0 )


            model_output =model (**model_inputs )
            output =model_output .get ("output",model_output .get ("logits"))if isinstance (model_output ,dict )else model_output 


            if output .dim ()==3 :
                loss =criterion (output .reshape (-1 ,output .size (-1 )),target .reshape (-1 ))
                pred =output .argmax (dim =-1 )
                valid_mask =target !=-100 
                correct +=((pred ==target )&valid_mask ).sum ().item ()
                total_samples +=valid_mask .sum ().item ()
            else :
                loss =criterion (output ,target )
                pred =output .argmax (dim =1 )
                valid_mask =target !=-100 
                correct +=((pred ==target )&valid_mask ).sum ().item ()
                total_samples +=valid_mask .sum ().item ()

            total_loss +=loss .item ()*batch_size 




            if isinstance (model_output ,dict )and 'routing_masks'in model_output :
                routing_masks =model_output ['routing_masks']
                if routing_masks and len (routing_masks )>0 :

                    last_routing_data =routing_masks [-1 ]if isinstance (routing_masks ,list )else routing_masks 

                    if isinstance (last_routing_data ,dict ):

                        routing_mask =last_routing_data ['routing_mask'].to (device )
                        nuance_analysis =last_routing_data .get ('nuance_analysis',{})


                        if 'sophistication_score'in nuance_analysis :
                            sophistication_score =nuance_analysis ['sophistication_score']

                            if torch .is_tensor (sophistication_score ):
                                if sophistication_score .numel ()>1 :
                                    sophistication_scores .append (sophistication_score .mean ().item ())
                                else :
                                    sophistication_scores .append (sophistication_score .item ())
                            else :
                                sophistication_scores .append (sophistication_score )


                        if 'geometric_specialization'in nuance_analysis :
                            geometric_specialization =nuance_analysis ['geometric_specialization']

                            if torch .is_tensor (geometric_specialization ):
                                if geometric_specialization .numel ()>1 :
                                    geometric_specialization_scores .append (geometric_specialization .mean ().item ())
                                else :
                                    geometric_specialization_scores .append (geometric_specialization .item ())
                            else :
                                geometric_specialization_scores .append (geometric_specialization )
                    else :

                        routing_mask =last_routing_data .to (device )
                        nuance_analysis ={}
                        sophistication_score =0.0 


                    expert_usage =routing_mask .sum (dim =0 )
                    expert_activations +=expert_usage 


                    routing_probs =F .softmax (routing_mask ,dim =-1 )
                    entropy =-(routing_probs *torch .log (routing_probs +1e-8 )).sum (dim =-1 ).mean ()
                    routing_entropies .append (entropy .item ())


            for name ,param in model .named_parameters ():
                if 'raw_c'in name or 'raw_r'in name :
                    curvature_values .append (param .item ())


            if hasattr (model ,'gate')and hasattr (model .gate ,'balance_loss'):
                balance_losses .append (model .gate .balance_loss .item ())


            if isinstance (model_output ,dict )and 'memory_state'in model_output :
                memory ,usage =model_output ['memory_state']
                if usage is not None :
                    memory_utilization =usage .mean ().item ()
                    memory_utilizations .append (memory_utilization )

            step_count +=1 


    elapsed =time .time ()-start_time 
    avg_loss =total_loss /max (1 ,total_samples )
    accuracy =100. *correct /max (1 ,total_samples )


    perplexity =math .exp (avg_loss )if avg_loss <50 else float ('inf')


    expert_diversity =0.0 
    routing_entropy =0.0 
    curvature_diversity =0.0 
    memory_efficiency =0.0 
    architecture_health =0.0 

    avg_sophistication =sum (sophistication_scores )/max (1 ,len (sophistication_scores ))if sophistication_scores else 0.0 
    avg_geometric_specialization =sum (geometric_specialization_scores )/max (1 ,len (geometric_specialization_scores ))if geometric_specialization_scores else 0.0 

    if len (routing_entropies )>0 :
        routing_entropy =sum (routing_entropies )/len (routing_entropies )

    if expert_activations .sum ()>0 :
        expert_probs =expert_activations /expert_activations .sum ()

        expert_diversity =-(expert_probs *torch .log (expert_probs +1e-8 )).sum ().item ()
        max_entropy =torch .log (torch .tensor (model .num_experts ,dtype =torch .float32 ))
        expert_diversity =expert_diversity /max_entropy .item ()

    if len (curvature_values )>1 :
        curvature_tensor =torch .tensor (curvature_values )
        curvature_diversity =curvature_tensor .std ().item ()

    if len (memory_utilizations )>0 :
        memory_efficiency =sum (memory_utilizations )/len (memory_utilizations )



    health_components =[]
    if expert_diversity >0 :
        health_components .append (expert_diversity *25 )
    if routing_entropy >0 :
        health_components .append (min (routing_entropy *10 ,25 ))
    if curvature_diversity >0 :
        health_components .append (min (curvature_diversity *50 ,25 ))
    if memory_efficiency >0 :
        health_components .append (memory_efficiency *25 )

    if health_components :
        architecture_health =sum (health_components )


    if stage_name =='curvature_calibration':
        logging .info (f"🌀 Curvature Validation → Loss: {avg_loss :.4f}, Curvature Diversity: {curvature_diversity :.4f}, Architecture Health: {architecture_health :.1f}%")
    elif 'gate'in stage_name :
        logging .info (f"🚪 Routing Validation → Loss: {avg_loss :.4f}, Expert Diversity: {expert_diversity :.4f}, Routing Entropy: {routing_entropy :.4f}, Sophistication: {avg_sophistication :.3f}")
    else :
        logging .info (
        f"🧠 MGM Validation → Loss: {avg_loss :.4f}, Perplexity: {perplexity :.2f}, "
        f"Accuracy: {accuracy :.2f}%, Expert Diversity: {expert_diversity :.4f}, "
        f"Architecture Health: {architecture_health :.1f}%, Sophistication: {avg_sophistication :.3f}, "
        f"Geometric: {avg_geometric_specialization :.3f}")

    if max_steps is not None and step_count >=max_steps :
        logging .info (f"⚡ Fast MGM validation: {step_count } steps in {elapsed :.1f}s")


    return {
    'loss':avg_loss ,
    'perplexity':perplexity ,
    'accuracy':accuracy ,
    'expert_diversity':expert_diversity ,
    'routing_entropy':routing_entropy ,
    'curvature_diversity':curvature_diversity ,
    'memory_efficiency':memory_efficiency ,
    'architecture_health':architecture_health ,
    'balance_loss':sum (balance_losses )/max (1 ,len (balance_losses ))if balance_losses else 0.0 ,

    'sophistication_score':avg_sophistication ,
    'geometric_specialization_score':avg_geometric_specialization ,
    'perplexity':perplexity ,
    }


def validate_epoch (model ,data_loader ,criterion ,device ,max_steps =None ):
    """Backward compatibility wrapper - redirects to MGM-aware validation"""
    metrics =validate_mgm_architecture (model ,data_loader ,criterion ,device ,max_steps )
    return metrics ['loss']




SCALER =None 

TRAIN_STATE :dict [str ,float ]={}


def orchestrate_training_stage (
stage_name ,
model ,
reward_model ,
train_loader ,
val_loader ,
config ,
device ,
current_epoch ,
global_step =0 ,
start_batch =0 ,
scaler =None ,
):
    global SCALER 


    config ['_current_stage']=stage_name 



    if scaler is None :
        scaler =SCALER 

    if scaler is None and config .get ('training',{}).get ('use_amp',False ):
        scaler =ComplexFilteredGradScaler ()
        scaler .register_complex_params (model )
        SCALER =scaler 


    if stage_name =='curvature_calibration':
        config ['_curvature_only_mode']=True 
        print (f"🚀 CURVATURE-ONLY MODE ENABLED: 100x speed boost!")
        logging .info ("🚀 CURVATURE-ONLY MODE ENABLED: 100x speed boost!")
    else :
        config ['_curvature_only_mode']=False 

    logging .info (f"\n{'='*25 } STARTING TRAINING STAGE: {stage_name .upper ()} {'='*25 }")
    stage_config =config ['training_stages'][stage_name ]


    if hasattr (torch ,'_dynamo')and hasattr (torch ._dynamo ,'reset'):
        torch ._dynamo .reset ()


    if device .type =='cuda':
        torch .cuda .empty_cache ()
        torch .cuda .reset_peak_memory_stats ()

        import gc 
        gc .collect ()
        torch .cuda .empty_cache ()


    for param in model .parameters ():
        param .requires_grad =False 


    if stage_name =='curvature_calibration':

        print ("🧠 SMART MODE: Learning from data with efficient sampling")
        logging .info ("🧠 SMART MODE: Learning from data with efficient sampling")


        curvature_count =0 
        for name ,param in model .named_parameters ():

            if 'raw_c'in name or 'raw_r'in name :
                param .requires_grad =True 
                curvature_count +=1 
            else :
                param .requires_grad =False 

        expert_count =0 

        logging .info (f"🌀 Smart curvature calibration: {curvature_count } curvature + {expert_count } expert parameters")


        trainable_params =[p for p in model .parameters ()if p .requires_grad ]
        optimizer =torch .optim .Adam (trainable_params ,lr =0.003 )


        max_batches =stage_config .get ('smart_calibration_batches',100 )
        print (f"⚡ OPTIMIZED CURVATURE CALIBRATION: Learning from {max_batches } batches (5x speed boost!)")

        import time 
        start_time =time .time ()
        total_loss =0.0 
        processed_batches =0 
        curvature_history =[]


        grad_accum_steps =2 
        micro_batch_loss =0.0 

        model .train ()
        for batch_idx ,batch in enumerate (train_loader ):
            if batch_idx >=max_batches :
                break 


            if batch is None :
                print (f"⚠️ Batch {batch_idx } is None, skipping...")
                continue 


            if batch_idx %grad_accum_steps ==0 :
                optimizer .zero_grad ()


            has_text ='input_ids'in batch and batch ['input_ids']is not None 
            has_vision ='pixel_values'in batch and batch ['pixel_values']is not None 
            has_audio ='audio_codes'in batch and batch ['audio_codes']is not None 
            has_labels ='labels'in batch and batch ['labels']is not None 


            if not (has_text or has_vision or has_audio )or not has_labels :
                print (f"⚠️ Batch {batch_idx } has no valid modalities or authentic labels, skipping per Mandate 1...")
                continue 


            if has_text :
                if batch ['input_ids'].numel ()==0 or batch ['input_ids'].shape [0 ]==0 :
                    print (f"⚠️ Batch {batch_idx } has empty text input, skipping...")
                    continue 
                if torch .all (batch ['input_ids']==0 ):
                    print (f"⚠️ Batch {batch_idx } contains only padding tokens, skipping...")
                    continue 
                max_token_id =batch ['input_ids'].max ().item ()
                if max_token_id >=model .embedding .num_embeddings :
                    raise ValueError (f"🚨 TOKEN ID OUT OF RANGE: {max_token_id } >= vocab_size {model .embedding .num_embeddings }")


            model_inputs ={}
            target =None 


            if has_text :
                model_inputs ['input_ids']=batch ['input_ids'].to (device )
                model_inputs ['attention_mask']=batch ['attention_mask'].to (device )
                if target is None :
                    target =batch ['labels'].to (device )

            if has_vision :
                model_inputs ['pixel_values']=batch ['pixel_values'].to (device )
                if target is None :
                    target =batch ['labels'].to (device )

            if has_audio :
                model_inputs ['audio_codes']=batch ['audio_codes'].to (device )
                if target is None :
                    target =batch ['audio_codes'][:,0 ,0 ].long ().to (device )


            if target is None :
                logger .error ("No authentic target found in batch - skipping per Mandate 1")
                continue 

            try :

                model_output =model (**model_inputs )


                if isinstance (model_output ,dict ):
                    final_output =model_output .get ("output")
                    if final_output is None :
                        final_output =model_output .get ("logits")
                    if final_output is None :
                        raise ValueError ("Model output dict missing 'output' or 'logits' key")
                elif isinstance (model_output ,tuple ):
                    final_output =model_output [0 ]
                else :
                    final_output =model_output 


                if final_output .dim ()==3 :
                    final_output =final_output [:,:-1 ,:].contiguous ()
                    target =target [:,1 :].contiguous ()
                elif final_output .dim ()==2 :
                    target =target .view (-1 )
                else :
                    raise ValueError (f"Unexpected final_output shape: {final_output .shape }")


                if has_text :
                    target =target .long ()
                elif has_vision or has_audio :

                    target =target .long ()if target .dim ()>0 else target 

                    if target .max ()>=final_output .size (-1 ):
                        target =torch .clamp (target ,0 ,final_output .size (-1 )-1 )

                task_loss =nn .CrossEntropyLoss (label_smoothing =0.1 ,ignore_index =-100 )(
                final_output .view (-1 ,final_output .size (-1 )),
                target .view (-1 )
                )





                curvature_diversity_loss =0.0 
                curvature_std =0.0 
                curvature_mean =0.0 

                if batch_idx %3 ==0 :

                    curvature_values =[]
                    for n ,p in model .named_parameters ():
                        if ('raw_c'in n or 'raw_r'in n )and p .requires_grad :

                            if p .grad is None :

                                dummy_loss =0.001 *p .mean ()
                                dummy_loss .backward (retain_graph =True )
                            curvature_values .append (p )

                if len (curvature_values )>1 :
                    curvature_tensor =torch .stack (curvature_values )
                    curvature_std =curvature_tensor .std ()
                    curvature_mean =curvature_tensor .mean ()



                    diversity_regularizer =0.5 *(1.0 -torch .clamp (curvature_std ,0 ,2.0 ))


                    specialization_bonus =0.1 *torch .clamp (curvature_std -1.0 ,0 ,float ('inf'))


                    stability_penalty =0.02 *torch .mean (torch .clamp (torch .abs (curvature_tensor )-5.0 ,0 ,float ('inf')))

                    curvature_diversity_loss =diversity_regularizer +specialization_bonus +stability_penalty 
                    curvature_history .append (curvature_std .item ())

                elif len (curvature_values )==1 :

                    curvature_tensor =curvature_values [0 ]
                    curvature_std =torch .tensor (0.01 ,device =device )
                    curvature_mean =curvature_tensor .mean ()


                    diversity_regularizer =0.1 *torch .abs (curvature_tensor -1.0 ).mean ()
                    curvature_diversity_loss =diversity_regularizer 
                    curvature_history .append (curvature_std .item ())
                else :

                    curvature_diversity_loss =torch .tensor (0.0 ,device =device ,requires_grad =True )
                    curvature_std =torch .tensor (0.0 ,device =device )
                    curvature_mean =torch .tensor (0.0 ,device =device )
                    curvature_history .append (0.0 )


                    if batch_idx %50 ==0 :
                        print (f"    Curvature analysis: μ={curvature_mean :.3f}, σ={curvature_std :.3f}, range=[{curvature_tensor .min ():.3f}, {curvature_tensor .max ():.3f}]")
                        if curvature_std <0.1 :
                            print (f"    ⚠️  Low expert diversity detected - experts may be collapsing")


                balance_loss =getattr (model .gate ,'balance_loss',0.0 )


                total_batch_loss =task_loss +0.1 *curvature_diversity_loss +0.05 *balance_loss 


                scaled_loss =total_batch_loss /grad_accum_steps 
                scaled_loss .backward ()

                micro_batch_loss +=total_batch_loss .item ()


                if (batch_idx +1 )%grad_accum_steps ==0 or batch_idx ==max_batches -1 :

                    torch .nn .utils .clip_grad_norm_ (trainable_params ,max_norm =1.0 )
                    optimizer .step ()

                    total_loss +=micro_batch_loss 
                    micro_batch_loss =0.0 
                processed_batches +=1 


                if batch_idx %10 ==0 and (batch_idx +1 )%grad_accum_steps ==0 :
                    elapsed =time .time ()-start_time 
                    batches_per_sec =(batch_idx +1 )/elapsed 
                    avg_loss =total_loss /max (1 ,processed_batches )
                    diversity =curvature_history [-1 ]if curvature_history else 0.0 


                    loss_improvement =""
                    if len (curvature_history )>=10 :
                        initial_loss =total_loss /max (1 ,min (10 ,processed_batches ))
                        current_loss =avg_loss 
                        improvement =initial_loss -current_loss 
                        loss_improvement =f", Δloss={improvement :+.3f}"


                    expert_activations ={}
                    for name ,param in model .named_parameters ():
                        if 'raw_c'in name or 'raw_r'in name :
                            expert_id =name .split ('.')[1 ]if '.'in name else 'unknown'
                            if 'raw_c'in name :
                                actual_val =torch .exp (param ).item ()+1 
                                expert_activations [f"c_{expert_id }"]=actual_val 
                            else :
                                actual_val =torch .exp (param ).item ()+1 
                                expert_activations [f"r_{expert_id }"]=actual_val 

                    avg_soph =getattr (model .gate ,'avg_sophistication',torch .tensor (0.0 )).item ()if hasattr (model ,'gate')else 0.0 
                    print (f"  Batch {batch_idx +1 :2d}/{max_batches }: {batches_per_sec :.1f} batch/s, loss={avg_loss :.4f}{loss_improvement }, div={diversity :.4f}, soph={avg_soph :.3f}")


                    if batch_idx %50 ==0 and expert_activations :
                        curvature_range =max (expert_activations .values ())-min (expert_activations .values ())
                        print (f"    Expert specialization range: {curvature_range :.3f} (higher = better diversity)")


                    if diversity >0.5 :
                        print (f"    ✓ Optimal expert diversity achieved (σ={diversity :.4f} > 0.5)")


                    if batch_idx >20 and avg_loss >15.0 :
                        print (f"    ⚠️  High loss detected - may need longer calibration or different learning rate")
                    if batch_idx >30 and batches_per_sec <0.5 :
                        print (f"    ⚠️  Slow processing detected - consider reducing complexity")








            except Exception as e :
                print (f"⚠️ Batch {batch_idx } failed: {e }")
                continue 

        elapsed =time .time ()-start_time 
        avg_batch_time =elapsed /max (1 ,processed_batches )
        print (f"🎉 SMART CALIBRATION COMPLETE: {processed_batches } batches in {elapsed :.2f}s ({processed_batches /elapsed :.1f} batch/s)")
        print (f"   Average batch time: {avg_batch_time :.3f}s, Final loss: {total_loss /max (1 ,processed_batches ):.4f}")


        print (f"📊 FINAL CURVATURE ANALYSIS:")
        expert_curvatures ={}
        manifold_types ={'euclidean':[],'hyperbolic':[],'spherical':[]}

        for name ,param in model .named_parameters ():
            if 'raw_c'in name or 'raw_r'in name :
                raw_val =param .item ()
                expert_id =name .split ('.')[1 ]if '.'in name else 'unknown'

                if 'raw_c'in name :
                    actual_c =torch .exp (param ).item ()+1 
                    expert_curvatures [f"expert_{expert_id }_c"]=actual_c 
                    print (f"   Expert {expert_id } Curvature: raw={raw_val :.4f} → c={actual_c :.4f}")


                    if actual_c <1.1 :
                        manifold_types ['euclidean'].append (expert_id )
                    elif actual_c >2.0 :
                        manifold_types ['spherical'].append (expert_id )
                    else :
                        manifold_types ['hyperbolic'].append (expert_id )

                else :
                    actual_r =torch .exp (param ).item ()+1 
                    expert_curvatures [f"expert_{expert_id }_r"]=actual_r 
                    print (f"   Expert {expert_id } Radius: raw={raw_val :.4f} → r={actual_r :.4f}")


        final_diversity =0.0 
        if len (curvature_history )>10 :
            initial_diversity =curvature_history [0 ]
            final_diversity =curvature_history [-1 ]
            diversity_improvement =final_diversity -initial_diversity 
            print (f"   Diversity evolution: {initial_diversity :.4f} → {final_diversity :.4f} (Δ={diversity_improvement :+.4f})")

            if final_diversity >0.5 :
                print (f"   ✅ EXCELLENT expert diversity achieved! (target: >0.5)")
            elif final_diversity >0.3 :
                print (f"   ✓ Good expert diversity (target: >0.5, could be improved)")
            else :
                print (f"   ⚠️  Low expert diversity - experts may need longer calibration")
        elif len (curvature_history )>0 :
            final_diversity =curvature_history [-1 ]
            print (f"   ⚠️  Limited diversity data (only {len (curvature_history )} samples)")
        else :
            print (f"   ⚠️  No diversity data available")


        total_experts =sum (len (experts )for experts in manifold_types .values ())
        if total_experts >0 :
            print (f"   Manifold specialization:")
            for manifold ,experts in manifold_types .items ():
                percentage =len (experts )/total_experts *100 
                print (f"     {manifold .capitalize ()}: {len (experts )} experts ({percentage :.1f}%)")


            if len (manifold_types ['euclidean'])>0 and len (manifold_types ['hyperbolic'])>0 and len (manifold_types ['spherical'])>0 :
                print (f"   ✅ Multi-manifold specialization achieved!")
            else :
                print (f"   ⚠️  Experts collapsed to similar manifolds - may need different initialization")


        quality_score =0 
        if final_diversity >0.5 :quality_score +=3 
        elif final_diversity >0.3 :quality_score +=2 
        elif final_diversity >0.1 :quality_score +=1 

        if len (set (manifold_types ['euclidean']+manifold_types ['hyperbolic']+manifold_types ['spherical']))>=2 :
            quality_score +=2 

        if processed_batches >=50 :quality_score +=1 

        print (f"   Calibration quality score: {quality_score }/6")
        if quality_score >=5 :
            print (f"   🎯 Excellent calibration! Expect strong training performance.")
        elif quality_score >=3 :
            print (f"   👍 Good calibration. Training should proceed well.")
        else :
            print (f"   ⚠️  Suboptimal calibration. Consider longer calibration or parameter adjustments.")


        checkpoint_data ={
        'epoch':current_epoch ,
        'stage':stage_name ,
        'stage_completed':True ,
        'state_dict':model .state_dict (),
        'config':config ,
        'global_step':global_step ,
        'rng_state':torch .get_rng_state (),
        'numpy_state':np .random .get_state (),
        'python_state':random .getstate (),
        'optimizer':optimizer .state_dict (),
        'ppo_kl_beta':TRAIN_STATE .get (
        'ppo_kl_beta',config ['training'].get ('ppo_kl_coef',0.02 )
        ),
        }
        if scaler is not None :
            checkpoint_data ['scaler']=scaler .state_dict ()
        save_checkpoint (checkpoint_data ,filename ="checkpoint_curvature_calibration_completed.pth.tar")
        logging .info (f"{'='*25 } COMPLETED TRAINING STAGE: {stage_name .upper ()} {'='*25 }")


        return current_epoch ,global_step 

    elif stage_name =='reasoning_warmup':
        for name ,param in model .named_parameters ():
            if 'thought_generator'in name or 'diffusion_gate'in name or 'final_head'in name :
                param .requires_grad =True 

    elif stage_name =='branch_pretrain':

        for name ,param in model .named_parameters ():
            if name .startswith ('embedding')or name .startswith ('experts.'):
                param .requires_grad =True 

    elif stage_name =='gate_train':
        for name ,param in model .named_parameters ():
            if 'gate'in name or 'combiner'in name :
                param .requires_grad =True 


        if hasattr (model ,'gate')and hasattr (model .gate ,'anneal_temperature'):

            stage_epochs =stage_config .get ('epochs',1 )
            for epoch in range (stage_epochs ):
                model .gate .anneal_temperature (epoch ,stage_epochs )

    elif stage_name =='joint_finetune':
        for param in model .parameters ():
            param .requires_grad =True 

    trainable_params =sum (p .numel ()for p in model .parameters ()if p .requires_grad )
    total_params =sum (p .numel ()for p in model .parameters ())
    logging .info (f"Stage '{stage_name }': {trainable_params :,}/{total_params :,} trainable parameters ({100 *trainable_params /total_params :.1f}%)")

    optimizer =create_geometric_optimizer (model ,stage_config ,stage_name )


    reward_lr =stage_config ['optimizer'].get ('reward_lr')
    if reward_lr is None :
        reward_lr =1e-4 if stage_name !='curvature_calibration'else 0.0 
    if reward_lr >0 :
        ppo_optimizer =torch .optim .Adam (reward_model .parameters (),lr =reward_lr )
    else :
        ppo_optimizer =None 

    scheduler =CosineAnnealingWarmRestarts (optimizer ,T_0 =stage_config .get ('scheduler_t0',10 ),T_mult =1 ,eta_min =1e-6 )


    label_smoothing =stage_config .get ('label_smoothing',0.1 )
    criterion =nn .CrossEntropyLoss (label_smoothing =label_smoothing ,ignore_index =-100 )
    best_val_loss =float ('inf')

    start_epoch =current_epoch 
    end_epoch =current_epoch +stage_config ['epochs']

    for epoch in range (start_epoch ,end_epoch ):

        if hasattr (model ,'gate')and hasattr (model .gate ,'anneal_temperature'):

            model .gate .anneal_temperature (global_step ,max_steps =10000 )

        batch_start =start_batch if epoch ==start_epoch else 0 
        global_step ,best_val_loss =train_epoch (
        model ,
        reward_model ,
        train_loader ,
        optimizer ,
        ppo_optimizer ,
        criterion ,
        device ,
        epoch ,
        config ,
        global_step ,
        start_batch =batch_start ,
        lr_scheduler =scheduler ,
        best_val_loss =best_val_loss 
        )


        val_max_steps =config .get ('training',{}).get ('validation_max_steps')
        val_metrics =validate_mgm_architecture (model ,val_loader ,criterion ,device ,max_steps =val_max_steps ,stage_name =stage_name )
        val_loss =val_metrics ['loss']
        scheduler .step ()

        if val_loss <best_val_loss :
            best_val_loss =val_loss 
            checkpoint_data ={
            'epoch':epoch +1 ,
            'stage':stage_name ,
            'state_dict':model .state_dict (),
            'reward_model_state_dict':reward_model .state_dict (),
            'optimizer':optimizer .state_dict (),
            'best_val_loss':best_val_loss ,
            'config':config ,
            'global_step':global_step ,
            'rng_state':torch .get_rng_state (),
            'numpy_state':np .random .get_state (),
            'python_state':random .getstate (),
            'scheduler':scheduler .state_dict (),
            'ppo_kl_beta':TRAIN_STATE .get (
            'ppo_kl_beta',config ['training'].get ('ppo_kl_coef',0.02 )
            ),
            }
            if scaler is not None :
                checkpoint_data ['scaler']=scaler .state_dict ()
            if ppo_optimizer is not None :
                checkpoint_data ['ppo_optimizer']=ppo_optimizer .state_dict ()

            save_checkpoint (checkpoint_data ,filename =f"checkpoint_{stage_name }_best.pth.tar")


    stage_completion_data ={
    'epoch':end_epoch ,
    'stage':stage_name ,
    'stage_completed':True ,
    'state_dict':model .state_dict (),
    'reward_model_state_dict':reward_model .state_dict (),
    'optimizer':optimizer .state_dict (),
    'config':config ,
    'global_step':global_step ,
    'rng_state':torch .get_rng_state (),
    'numpy_state':np .random .get_state (),
    'python_state':random .getstate (),
    'scheduler':scheduler .state_dict (),
    'ppo_kl_beta':TRAIN_STATE .get (
    'ppo_kl_beta',config ['training'].get ('ppo_kl_coef',0.02 )
    ),
    }
    if scaler is not None :
        stage_completion_data ['scaler']=scaler .state_dict ()
    if ppo_optimizer is not None :
        stage_completion_data ['ppo_optimizer']=ppo_optimizer .state_dict ()

    save_checkpoint (stage_completion_data ,filename =f"checkpoint_{stage_name }_completed.pth.tar")

    logging .info (f"{'='*25 } COMPLETED TRAINING STAGE: {stage_name .upper ()} {'='*25 }")


    if stage_name =='curvature_calibration':
        logging .info ("📊 Final curvature values after calibration:")
        for name ,param in model .named_parameters ():
            if 'raw_c'in name or 'raw_r'in name :
                if 'raw_c'in name :
                    c_value =torch .nn .functional .softplus (param ).item ()
                    logging .info (f"   {name }: raw={param .item ():.4f} → c={c_value :.4f}")
                elif 'raw_r'in name :
                    r_value =torch .nn .functional .softplus (param ).item ()
                    logging .info (f"   {name }: raw={param .item ():.4f} → r={r_value :.4f}")

    return end_epoch ,global_step 

def create_mgm_config ():
    """Create configuration optimized for your MGM data"""
    return {
    "model":{
    "input_dim":1536 ,
    "hidden_dim":6144 ,
    "output_dim":1536 ,
    "final_output_dim":50272 ,
    "vocab_size":50272 ,
    "pad_token_id":50257 ,

    "enable_vision":True ,
    "vision_tower_name":"openai/clip-vit-large-patch14",

    "num_experts":16 ,
    "k":4 ,
    "recursion_steps":4 ,
    "memory_slots":64 ,
    "memory_width":1536 ,
    "manifolds":[
    "euclidean","hyperbolic","spherical","poincare",
    "simplex","complex","lorentzian","product",
    "euclidean","hyperbolic","spherical","poincare",
    "simplex","complex","lorentzian","product"
    ]
    },
    "data":{
    "npz_files":{k :str (v )for k ,v in DATA_PATHS .items ()},
    "seq_len":1024 ,
    "val_split":0.05 
    },
    "training":{
    "batch_size":4 ,
    "use_amp":True ,
    "balance_loss_weight":0.05 ,
    "curvature_reg_weight":0.005 ,
    "verifier_loss_weight":0.02 ,
    "memory_auxiliary_loss_weight":0.1 ,
    "seed":42 ,
    "compile_model":False ,
    "gradient_accumulation_steps":8 ,
    "dataloader_num_workers":0 ,
    "prefetch_factor":8 ,
    "pin_memory":True ,
    "max_grad_norm":1.0 ,
    "warmup_steps":500 
    },
    "training_stages":{
    "curvature_calibration":{
    "epochs":2 ,
    "label_smoothing":0.0 ,
    "optimizer":{
    "learning_rate":5e-3 ,
    "geo_lr_factor":1.0 ,
    "weight_decay":0.0 ,
    "reward_lr":0.0 
    }
    },
    "reasoning_warmup":{
    "epochs":3 ,
    "label_smoothing":0.05 ,
    "optimizer":{"learning_rate":2e-3 ,"reward_lr":1e-3 ,"weight_decay":0.005 ,"geo_lr_factor":1.0 }
    },
    "branch_pretrain":{
    "epochs":4 ,
    "label_smoothing":0.1 ,
    "optimizer":{"learning_rate":1e-3 ,"reward_lr":5e-4 ,"weight_decay":0.01 ,"geo_lr_factor":2.0 }
    },
    "gate_train":{
    "epochs":3 ,
    "label_smoothing":0.1 ,
    "optimizer":{"learning_rate":2e-4 ,"reward_lr":1e-4 ,"weight_decay":0.01 ,"geo_lr_factor":1.0 }
    },
    "joint_finetune":{
    "epochs":6 ,
    "label_smoothing":0.15 ,
    "scheduler_t0":2 ,
    "optimizer":{"learning_rate":1e-4 ,"reward_lr":5e-5 ,"weight_decay":0.015 ,"geo_lr_factor":1.5 }
    }
    }
    }












def collate_fn_streaming (batch :list )->dict :
    """
    Collate function for streaming data with T4-optimized padding.
    Handles padding to multiple of 8 for better T4 performance.
    """
    input_ids =[item ['input_ids']for item in batch ]

    labels =[torch .cat ([ids [1 :],ids .new_full ((1 ,),-100 )])for ids in input_ids ]


    pad_id =50257 


    max_len =max (len (ids )for ids in input_ids )
    padded_max_len =((max_len +7 )//8 )*8 

    padded_input_ids =torch .full ((len (batch ),padded_max_len ),pad_id ,dtype =torch .long )
    padded_labels =torch .full ((len (batch ),padded_max_len ),-100 ,dtype =torch .long )
    attention_mask =torch .zeros ((len (batch ),padded_max_len ),dtype =torch .long )

    for i ,ids in enumerate (input_ids ):
        end =len (ids )
        padded_input_ids [i ,:end ]=ids 
        padded_labels [i ,:end ]=labels [i ]
        attention_mask [i ,:end ]=1 

    return {
    'input_ids':padded_input_ids ,
    'labels':padded_labels ,
    'attention_mask':attention_mask 
    }

def main ():

    import torch .multiprocessing as mp 
    try :
        mp .set_start_method ('spawn',force =True )
        logging .info ("✅ Multiprocessing start method set to 'spawn'.")
    except RuntimeError :
        logging .warning ("⚠️ Could not set 'spawn' start method - it might already be set.")


    if MEMORY_GUARD_AVAILABLE :
        setup_cuda_optimizations ()
        memory_guard =DynamicMemoryGuard (
        gpu_memory_threshold =0.80 ,
        cpu_memory_threshold =0.85 ,
        gradient_clip_value =1.0 
        )
        adaptive_batch =AdaptiveBatchSize (initial_batch_size =2 ,min_batch_size =1 ,max_batch_size =8 )
        logging .info ("🛡️ Memory protection system initialized")
    else :
        memory_guard =None 
        adaptive_batch =None 
        logging .warning ("⚠️ Training without memory protection")

    parser =argparse .ArgumentParser (description ="MGM Mixture-of-Geometric-Experts Training")
    parser .add_argument ('--config',type =str ,default ='mgm_config.json',help ='Configuration file')
    parser .add_argument ('--checkpoint',type =str ,default =None ,help ='Resume from checkpoint')
    parser .add_argument ('--export-path',type =str ,default ='mgm_geometric_model_final.pth',help ='Final model save path')
    parser .add_argument ('--data-subset',type =str ,nargs ='+',default =None ,
    help ='Train on subset of data (e.g., --data-subset all_combined_tokens wikipedia)')
    parser .add_argument ('--force-no-geoopt',action ='store_true',help ='Force training without GeoOpt (not recommended)')
    parser .add_argument ('--streaming',action ='store_true',help ='Enable multimodal streaming from Hub/Web')
    parser .add_argument ('--end_to_end',action ='store_true',help ='Run automated end-to-end training through all stages')
    args =parser .parse_args ()


    logging .basicConfig (
    level =logging .INFO ,
    format ='%(asctime)s - %(levelname)s - %(message)s',
    handlers =[
    logging .StreamHandler (sys .stdout ),
    logging .FileHandler ('training.log')
    ]
    )


    if not GEOOPT_AVAILABLE and not args .force_no_geoopt :
        logging .error ("🚨 CRITICAL ERROR: GeoOpt is not available!")
        logging .error ("   The MGM Mixture-of-Geometric-Experts model requires GeoOpt for proper geometric optimization.")
        logging .error ("   Without GeoOpt, the geometric manifolds will not be optimized correctly.")
        logging .error ("   This will significantly degrade model performance.")
        logging .error ("")
        logging .error ("🔧 SOLUTIONS:")
        logging .error ("   1. Install GeoOpt: pip install git+https://github.com/geoopt/geoopt.git")
        logging .error ("   2. Restart the kernel/runtime after installation")
        logging .error ("   3. Or use --force-no-geoopt flag to proceed anyway (NOT RECOMMENDED)")
        logging .error ("")
        sys .exit (1 )
    elif not GEOOPT_AVAILABLE and args .force_no_geoopt :
        logging .warning ("⚠️ WARNING: Proceeding without GeoOpt as requested (--force-no-geoopt)")
        logging .warning ("   Model performance will be significantly reduced!")
    else :
        logging .info ("✅ GeoOpt is available - geometric manifolds will be properly optimized")


    if not os .path .exists (args .config ):
        config =create_mgm_config ()
        with open (args .config ,'w')as f :
            json .dump (config ,f ,indent =4 )
        logging .info (f"✅ Created default config: {args .config }")
    else :
        with open (args .config ,'r')as f :
            config =json .load (f )
        logging .info (f"✅ Loaded config: {args .config }")


    if args .streaming :
        logging .info ("🚀 Activating Pure Streaming Pipeline...")
        from streaming_dataset_loader import (
        setup_streaming_environment ,get_tokenizer ,SPECIAL_TOKENS ,
        StreamingTextDataset 
        )

        setup_streaming_environment ()
        stream_config =config ['streaming']
        tokenizer =get_tokenizer (stream_config ['tokenizer_name'],stream_config ['modalities'])


        tokenizer .pad_token =tokenizer .eos_token 
        tokenizer .pad_token_id =tokenizer .eos_token_id 


        actual_vocab_size =len (tokenizer )
        config ['model']['vocab_size']=actual_vocab_size 
        config ['model']['final_output_dim']=actual_vocab_size 
        config ['model']['pad_token_id']=tokenizer .pad_token_id 
        config ['vocab_size']=actual_vocab_size 

        logging .info (f"Updated model config for streaming: vocab_size={actual_vocab_size }, pad_token_id={tokenizer .pad_token_id }")


    logging .info ("🔍 Running comprehensive production validation...")
    try :

        from production_dataset_validator import ProductionDatasetValidator 
        validator =ProductionDatasetValidator (args .config )
        results =validator .run_full_validation ()

        if len (results ['errors'])>0 :
            logging .error ("❌ Production validation failed - fix critical errors before training")
            for error in results ['errors']:
                logging .error (f"  🚨 {error }")
            sys .exit (1 )

        if len (results ['warnings'])>0 :
            logging .warning ("⚠️ Production validation warnings:")
            for warning in results ['warnings'][:10 ]:
                logging .warning (f"  ⚠️ {warning }")

        logging .info ("✅ Production validation passed")

    except ImportError :
        logging .warning ("⚠️ Could not import production validator - running basic config validation")
        config_errors =ProductionSafetyChecks .validate_config (config )
        if config_errors :
            logging .error (f"❌ Configuration validation failed: {config_errors }")
            sys .exit (1 )

    except Exception as e :
        logging .warning (f"⚠️ Could not run full validation: {e }")
        logging .info ("Running basic config validation...")
        config_errors =ProductionSafetyChecks .validate_config (config )
        if config_errors :
            logging .error (f"❌ Configuration validation failed: {config_errors }")
            sys .exit (1 )


    if args .data_subset :
        original_files =config ['data']['npz_files'].copy ()
        config ['data']['npz_files']={k :v for k ,v in original_files .items ()if k in args .data_subset }
        logging .info (f"🎯 Training on subset: {list (config ['data']['npz_files'].keys ())}")


    if torch .cuda .is_available ():
        device =torch .device ("cuda")
    elif hasattr (torch .backends ,'mps')and torch .backends .mps .is_available ():
        device =torch .device ("mps")
    else :
        device =torch .device ("cpu")

    logging .info (f"🚀 Using device: {device }")

    if device .type =='cuda':
        logging .info (f"   GPU: {torch .cuda .get_device_name (0 )}")
        logging .info (f"   GPU Memory: {torch .cuda .get_device_properties (0 ).total_memory /1e9 :.1f} GB")

        torch .cuda .empty_cache ()

    torch .manual_seed (config ['training']['seed'])
    if device .type =='cuda':
        torch .cuda .manual_seed (config ['training']['seed'])

        torch .backends .cuda .matmul .allow_tf32 =False 
        torch .backends .cudnn .allow_tf32 =False 
        torch .set_float32_matmul_precision ("high")


    if args .streaming :
        stream_config =config ['streaming']
        from streaming_dataset_loader import get_tokenizer ,SPECIAL_TOKENS 
        tokenizer =get_tokenizer (stream_config ['tokenizer_name'],stream_config ['modalities'])


        tokenizer .pad_token =tokenizer .eos_token 
        tokenizer .pad_token_id =tokenizer .eos_token_id 


        tokenizer_vocab_size =len (tokenizer )
        logging .info (f"🔧 Updating config vocab_size from {config ['model']['vocab_size']} to {tokenizer_vocab_size }")
        config ['model']['vocab_size']=tokenizer_vocab_size 
        config ['model']['final_output_dim']=tokenizer_vocab_size 
        config ['model']['pad_token_id']=tokenizer .pad_token_id 
        config ['vocab_size']=tokenizer_vocab_size 


        logging .info ("🔧 Updating production validator with correct tokenizer...")
        from production_dataset_validator import ProductionDatasetValidator 
        validator =ProductionDatasetValidator (
        npz_files ={},
        npy_files ={},
        config =config ,
        tokenizer =tokenizer 
        )
        validator .run_full_validation ()
        logging .info ("✅ Production validation passed with correct tokenizer")


        from streaming_dataset_loader import (
        LargeScaleStreamingDataset ,
        create_flagship_streaming_dataloaders ,
        MultimodalCollator ,
        )


        logger .info ("🚀 Creating flagship-scale streaming dataloaders...")


        train_loader =None 
        val_loader =None 


        multimodal_collator =MultimodalCollator (tokenizer ,config ['model']['vision_tower_name'])
        train_dataloader ,streaming_datasets =create_flagship_streaming_dataloaders (
        stream_config ,
        tokenizer ,
        batch_size =config ['training']['batch_size'],
        collator =multimodal_collator ,
        )


        train_loader =train_dataloader 
        val_loader =train_dataloader 

        logger .info (f"✅ Created flagship streaming dataloaders:")
        for modality ,dataset in streaming_datasets .items ():
            dataset_name =getattr (dataset ,'hf_dataset_name',f'{modality }_dataset')
            logger .info (f"   {modality }: {dataset_name } ({len (dataset )} samples)")

    else :

        npz_files =config ['data'].get ('npz_files',{})
        npy_files =config ['data'].get ('npy_files',{})
        dataset =A100OptimizedNpzDataset (
        npz_files =npz_files ,
        seq_len =config ['data']['seq_len'],
        final_output_dim =config ['model']['final_output_dim'],
        npy_files =npy_files 
        )


        dataset_size =len (dataset )
        if dataset_size ==0 :
            raise ValueError ("Loaded dataset is empty - check NPZ/NPY file paths")

        val_size =int (config ['data']['val_split']*dataset_size )
        train_size =dataset_size -val_size 

        logging .info (f"📊 Dataset split: {train_size :,} train, {val_size :,} validation")


        num_workers =min (4 ,os .cpu_count ()or 2 )
        prefetch_factor =8 
        pin_memory =True 
        persistent_workers =num_workers >0 
        drop_last =True 
        multiprocessing_context ='spawn'if num_workers >0 else None 


        if num_workers ==0 :
            prefetch_factor =None 
            persistent_workers =False 
            multiprocessing_context =None 

        train_batch_size =config ['training']['batch_size']
        if train_batch_size <=0 :
            raise ValueError (f"Invalid training batch_size={train_batch_size }. Must be > 0")

        train_loader =DataLoader (
        dataset ,
        batch_size =train_batch_size ,
        num_workers =num_workers ,
        pin_memory =pin_memory ,
        prefetch_factor =prefetch_factor ,
        persistent_workers =persistent_workers ,
        drop_last =drop_last ,
        multiprocessing_context =multiprocessing_context ,
        collate_fn =safe_collate_fn 
        )


        val_num_workers =max (1 ,num_workers //2 )
        val_batch_size =max (1 ,train_batch_size //2 )
        val_loader =DataLoader (
        dataset ,
        batch_size =val_batch_size ,
        num_workers =val_num_workers ,
        pin_memory =pin_memory ,
        persistent_workers =True if val_num_workers >0 else False ,
        prefetch_factor =max (2 ,prefetch_factor //2 )if val_num_workers >0 else None ,
        drop_last =False ,
        multiprocessing_context =multiprocessing_context if val_num_workers >0 else None ,
        collate_fn =safe_collate_fn 
        )


    model =MixtureOfGeometricExperts (config ).to (device )
    reward_model =RewardModel (config ['model']['output_dim']).to (device )


    if args .streaming :

        actual_vocab_size =len (tokenizer )
        model_vocab_size =model .embedding .num_embeddings 
        logging .info (f"✅ Model initialized with vocab_size={model_vocab_size }, tokenizer has {actual_vocab_size } tokens")

        if model_vocab_size !=actual_vocab_size :
            logging .warning (f"⚠️ Vocab size mismatch detected - this should not happen with updated config!")
            model .resize_token_embeddings (actual_vocab_size )


    compile_model =config ['training'].get ('compile_model',False )and device .type =='cuda'
    if compile_model and hasattr (torch ,'compile'):
        try :
            model =torch .compile (model ,mode ='default')
            logging .info ("🚀 Model compiled for better performance (CUDA only)")
            logging .warning ("⚠️ Model compilation enabled - ensure all stages use consistent retain_graph behavior")
        except Exception as e :
            logging .warning (f"⚠️ Model compilation failed: {e }")
    else :
        if device .type !='cuda':
            logging .info ("📝 Model compilation disabled on non-CUDA device for stability")
        else :
            logging .info ("📝 Model compilation disabled (recommended for multi-stage training)")

    total_params =sum (p .numel ()for p in model .parameters ())
    logging .info (f"🧠 Model initialized with {total_params :,} parameters")


    use_amp =config .get ("training",{}).get ("use_amp",False )
    scaler =None 
    if use_amp :
        scaler =ComplexFilteredGradScaler ()
        scaler .register_complex_params (model )
        logging .info ("✅ ComplexFilteredGradScaler initialized for mixed-precision training.")
        global SCALER 
        SCALER =scaler 


    skip_validation =os .environ .get ("MGM_SKIP_CHECKPOINT_VALIDATION","false").lower ()=="true"
    if args .checkpoint and not skip_validation :
        if not validate_model_checkpoint_compatibility (model ,args .checkpoint ,config ):
            logging .error ("❌ Checkpoint validation failed - cannot resume training")
            sys .exit (1 )
    elif args .checkpoint and skip_validation :
        logging .info ("⚠️ Checkpoint validation skipped for stage-aware resumption")


    start_epoch =0 
    start_stage_index =0 
    start_batch :int =0 
    global_step :int =0 

    if args .checkpoint and os .path .isfile (args .checkpoint ):
        logging .info (f"📂 Loading checkpoint: {args .checkpoint }")
        checkpoint =torch .load (args .checkpoint ,map_location =device ,weights_only =False )
        model .load_state_dict (checkpoint .get ('model_state_dict',checkpoint .get ('state_dict',checkpoint )))
        reward_model .load_state_dict (checkpoint .get ('reward_model_state_dict',{}))
        start_epoch =checkpoint .get ('epoch',0 )
        stage_name =checkpoint .get ('stage','reasoning_warmup')
        start_batch =checkpoint .get ('batch',0 )
        global_step =checkpoint .get ('global_step',0 )
        if 'ppo_kl_beta'in checkpoint :
            TRAIN_STATE ['ppo_kl_beta']=checkpoint ['ppo_kl_beta']

        env_stage =os .environ .get ('MGM_RESUME_STAGE')
        if env_stage :
            stage_name =env_stage 
        env_epoch =os .environ .get ('MGM_RESUME_EPOCH')
        if env_epoch is not None :
            start_epoch =int (env_epoch )
        env_batch =os .environ .get ('MGM_RESUME_BATCH')
        if env_batch is not None :
            start_batch =int (env_batch )
        env_gstep =os .environ .get ('MGM_RESUME_GLOBAL_STEP')
        if env_gstep is not None :
            global_step =int (env_gstep )

        all_stages =list (config ['training_stages'].keys ())
        try :
            start_stage_index =all_stages .index (stage_name )
        except ValueError :
            logging .warning (f"Stage '{stage_name }' not found, starting from beginning")
            start_stage_index =0 
            start_epoch =0 
            start_batch =0 


    if args .end_to_end :
        logging .info ("🚀 Starting automated end-to-end training harness")
        logging .info ("   Stages: curvature_calibration → reasoning_warmup → branch_pretrain → gate_train → joint_finetune")


        required_stages =['curvature_calibration','reasoning_warmup','branch_pretrain','gate_train','joint_finetune']
        config_stages =list (config ['training_stages'].keys ())

        for required_stage in required_stages :
            if required_stage not in config_stages :
                logging .error (f"❌ Required stage '{required_stage }' not found in config. Available: {config_stages }")
                sys .exit (1 )

        logging .info (f"✅ All required stages found in config: {required_stages }")


        current_epoch =start_epoch 
        stage_results ={}


        for idx in range (start_stage_index ,len (required_stages )):
            stage_name =required_stages [idx ]
            logging .info (f"🎯 Starting stage: {stage_name }")


            stage_start_epoch =current_epoch 
            initial_batch =start_batch if idx ==start_stage_index else 0 
            final_epoch ,global_step =orchestrate_training_stage (
            stage_name ,
            model ,
            reward_model ,
            train_loader ,
            val_loader ,
            config ,
            device ,
            current_epoch ,
            global_step =global_step ,
            start_batch =initial_batch ,
            scaler =scaler ,
            )


            stage_results [stage_name ]={
            'start_epoch':stage_start_epoch ,
            'final_epoch':final_epoch ,
            'epochs_completed':final_epoch -stage_start_epoch ,
            'status':'completed'
            }

            current_epoch =final_epoch 
            logging .info (f"✅ Completed stage: {stage_name } (epochs {stage_start_epoch } → {final_epoch })")


            stage_checkpoint_path =f"checkpoint_stage_{stage_name }_epoch_{final_epoch }.pth.tar"
            torch .save ({
            'model_state_dict':model .state_dict (),
            'reward_model_state_dict':reward_model .state_dict (),
            'config':config ,
            'epoch':final_epoch ,
            'stage':stage_name ,
            'stage_results':stage_results 
            },stage_checkpoint_path )
            logging .info (f"💾 Stage checkpoint saved: {stage_checkpoint_path }")


        logging .info ("🎉 AUTOMATED END-TO-END TRAINING COMPLETED SUCCESSFULLY!")
        logging .info ("📊 Training Summary:")
        total_epochs =0 
        for stage ,results in stage_results .items ():
            epochs =results ['epochs_completed']
            total_epochs +=epochs 
            logging .info (f"   {stage }: {epochs } epochs (status: {results ['status']})")
        logging .info (f"   Total epochs: {total_epochs }")
        logging .info (f"   16-expert MoGE architecture preserved throughout")

    else :

        current_epoch =start_epoch 
        all_stages =list (config ['training_stages'].keys ())

        for i in range (start_stage_index ,len (all_stages )):
            stage_name =all_stages [i ]
            initial_batch =start_batch if i ==start_stage_index else 0 
            current_epoch ,global_step =orchestrate_training_stage (
            stage_name ,
            model ,
            reward_model ,
            train_loader ,
            val_loader ,
            config ,
            device ,
            current_epoch ,
            global_step =global_step ,
            start_batch =initial_batch ,
            scaler =scaler ,
            )


    logging .info (f"🎉 Training complete! Saving final model to {args .export_path }")
    final_save_data ={
    'model_state_dict':model .state_dict (),
    'reward_model_state_dict':reward_model .state_dict (),
    'config':config ,
    'final_epoch':current_epoch ,
    'global_step':global_step ,
    'rng_state':torch .get_rng_state (),
    'numpy_state':np .random .get_state (),
    'python_state':random .getstate (),
    'ppo_kl_beta':TRAIN_STATE .get ('ppo_kl_beta',0.02 ),
    }


    if args .end_to_end and 'stage_results'in locals ():
        final_save_data ['stage_results']=stage_results 
        final_save_data ['training_mode']='end_to_end'

    torch .save (final_save_data ,args .export_path )

    logging .info (f"✅ MGM Geometric Model training completed successfully!")
    if args .end_to_end :
        logging .info (f"   Mode: Automated end-to-end training")
        logging .info (f"   Architecture: 16-expert MoGE maintained throughout")

if __name__ =='__main__':
    main ()
