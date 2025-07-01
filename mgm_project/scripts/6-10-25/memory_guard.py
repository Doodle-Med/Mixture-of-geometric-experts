#!/usr/bin/env python3


"""
Dynamic Memory Guard and Numerical Stability System
Prevents OOM errors and numerical instabilities during training
"""

import torch 
import gc 
import psutil 
import warnings 
import sys 
from contextlib import contextmanager 
from typing import Optional ,Dict ,Any 
import logging 

logger =logging .getLogger (__name__ )

class DynamicMemoryGuard :
    """Real-time memory monitoring and protection system"""

    def __init__ (self ,
    gpu_memory_threshold :float =0.85 ,
    cpu_memory_threshold :float =0.90 ,
    auto_cleanup :bool =True ,
    gradient_clip_value :float =1.0 ):
        self .gpu_threshold =gpu_memory_threshold 
        self .cpu_threshold =cpu_memory_threshold 
        self .auto_cleanup =auto_cleanup 
        self .gradient_clip_value =gradient_clip_value 
        self .memory_history =[]


        if torch .cuda .is_available ():
            torch .cuda .empty_cache ()

            torch .backends .cudnn .benchmark =False 
            torch .backends .cudnn .deterministic =True 

    def check_memory_status (self )->Dict [str ,float ]:
        """Check current memory usage"""
        status ={}


        if torch .cuda .is_available ():
            gpu_memory =torch .cuda .memory_allocated ()/torch .cuda .max_memory_allocated ()
            gpu_reserved =torch .cuda .memory_reserved ()/torch .cuda .max_memory_reserved ()
            status ['gpu_allocated']=gpu_memory 
            status ['gpu_reserved']=gpu_reserved 
            status ['gpu_free']=1.0 -gpu_reserved 


        cpu_percent =psutil .virtual_memory ().percent /100.0 
        status ['cpu_usage']=cpu_percent 
        status ['cpu_free']=1.0 -cpu_percent 

        return status 

    def emergency_cleanup (self ):
        """Emergency memory cleanup"""
        logger .warning ("🚨 Emergency memory cleanup triggered")


        if torch .cuda .is_available ():
            torch .cuda .empty_cache ()
            torch .cuda .ipc_collect ()


        gc .collect ()


        sys .modules .clear ()

    def safe_forward_pass (self ,model ,batch_data ):
        """Safe forward pass with memory monitoring"""
        try :

            status =self .check_memory_status ()
            if status .get ('gpu_reserved',0 )>self .gpu_threshold :
                self .emergency_cleanup ()


            with torch .cuda .amp .autocast (enabled =True ):
                output =model (batch_data )

            return output 

        except RuntimeError as e :
            if "out of memory"in str (e ).lower ():
                logger .error (f"🚨 OOM Error: {e }")
                self .emergency_cleanup ()

                with torch .cuda .amp .autocast (enabled =True ):
                    return model (batch_data )
            else :
                raise 

    def safe_backward_pass (self ,loss ,model ):
        """Safe backward pass with gradient clipping"""
        try :

            loss .backward ()


            torch .nn .utils .clip_grad_norm_ (model .parameters (),self .gradient_clip_value )


            has_nan =False 
            for param in model .parameters ():
                if param .grad is not None and torch .isnan (param .grad ).any ():
                    has_nan =True 
                    break 

            if has_nan :
                logger .warning ("🚨 NaN gradients detected - zeroing gradients")
                model .zero_grad ()
                return False 

            return True 

        except RuntimeError as e :
            if "out of memory"in str (e ).lower ():
                logger .error (f"🚨 OOM during backward: {e }")
                self .emergency_cleanup ()
                model .zero_grad ()
                return False 
            else :
                raise 

@contextmanager 
def numerical_stability_context ():
    """Context manager for numerical stability"""

    original_anomaly =torch .is_anomaly_detection_enabled ()

    try :

        torch .autograd .set_detect_anomaly (True )


        warnings .filterwarnings ('ignore',category =UserWarning )

        yield 

    except Exception as e :
        logger .error (f"🚨 Numerical instability detected: {e }")

        if torch .cuda .is_available ():
            torch .cuda .empty_cache ()
        gc .collect ()
        raise 

    finally :

        torch .autograd .set_detect_anomaly (original_anomaly )

class AdaptiveBatchSize :
    """Dynamically adjust batch size based on memory usage"""

    def __init__ (self ,initial_batch_size :int =8 ,min_batch_size :int =1 ,max_batch_size :int =32 ):
        self .current_batch_size =initial_batch_size 
        self .min_batch_size =min_batch_size 
        self .max_batch_size =max_batch_size 
        self .success_count =0 
        self .failure_count =0 

    def adjust_on_success (self ):
        """Increase batch size on successful training steps"""
        self .success_count +=1 
        self .failure_count =0 


        if self .success_count >=10 and self .current_batch_size <self .max_batch_size :
            self .current_batch_size =min (self .current_batch_size *2 ,self .max_batch_size )
            self .success_count =0 
            logger .info (f"📈 Increased batch size to {self .current_batch_size }")

    def adjust_on_failure (self ):
        """Decrease batch size on OOM or numerical issues"""
        self .failure_count +=1 
        self .success_count =0 

        if self .current_batch_size >self .min_batch_size :
            self .current_batch_size =max (self .current_batch_size //2 ,self .min_batch_size )
            logger .warning (f"📉 Reduced batch size to {self .current_batch_size }")

    def get_batch_size (self )->int :
        return self .current_batch_size 

def setup_cuda_optimizations ():
    """Setup optimal CUDA settings for training"""
    if not torch .cuda .is_available ():
        return 


    torch .cuda .empty_cache ()


    import os 
    os .environ ['PYTORCH_CUDA_ALLOC_CONF']='expandable_segments:True,roundup_power2_divisions:16'


    torch .backends .cuda .matmul .allow_tf32 =True 
    torch .backends .cudnn .allow_tf32 =True 


    torch .backends .cudnn .benchmark =True 
    torch .backends .cudnn .deterministic =False 

    logger .info ("✅ CUDA optimizations configured")

def create_safe_optimizer (model_parameters ,lr =1e-4 ):
    """Create optimizer with numerical stability features"""
    return torch .optim .AdamW (
    model_parameters ,
    lr =lr ,
    betas =(0.9 ,0.999 ),
    eps =1e-8 ,
    weight_decay =0.01 ,
    amsgrad =True 
    )
