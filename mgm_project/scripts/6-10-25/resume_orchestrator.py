#!/usr/bin/env python3


"""Stage-aware resume launcher for train_geometric_model_v2.py.

This helper downloads a checkpoint if required and injects resume logic
without modifying the original trainer script."""

import argparse 
import json 
import os 
import sys 
from pathlib import Path 
import threading 
import time 
import logging 
from typing import List ,Tuple ,Optional 


import train_geometric_model_v2 as trainer 
from resume_helper import (
train_epoch_with_resume ,
load_and_validate_checkpoint ,
patch_trainer_for_resume ,
)


original_train_epoch =trainer .train_epoch 


DEFAULT_CHECKPOINT_URL =(
"https://huggingface.co/doodle-med/mgm-checkpoints/resolve/main/checkpoint_epoch_0_batch_165000.pth.tar"
)






def _extract_metadata (path :Path )->Tuple [str ,int ,int ]:
    """Return stage, epoch and batch stored in a checkpoint file or inferred from its filename."""
    import torch 
    import re 

    stage =""
    epoch =0 
    batch =0 

    try :
        ckpt =torch .load (str (path ),map_location ="cpu",weights_only =False )
        stage =ckpt .get ("stage","")
        epoch =int (ckpt .get ("epoch",0 ))
        batch =int (ckpt .get ("batch",0 ))
    except Exception :
        ckpt ={}

    if not stage :


        m =re .search (r"checkpoint_([a-zA-Z0-9_]+)",path .name )
        if m :
            stage =m .group (1 )

    if epoch ==0 and batch ==0 :
        m =re .search (r"epoch_(\d+)_batch_(\d+)",path .name )
        if m :
            epoch =int (m .group (1 ))
            batch =int (m .group (2 ))

    return stage ,epoch ,batch 


def _stage_order (config :dict )->List [str ]:
    stages =config .get ("training_stages",{})
    if isinstance (stages ,dict ):
        return list (stages .keys ())
    return []


def _find_latest_checkpoint (directory :Path ,order :List [str ])->Optional [Tuple [Path ,str ]]:
    """Return path and stage for the most advanced checkpoint in *directory*."""
    best =None 
    best_key =(-1 ,-1 ,-1 )
    for ckpt in directory .glob ("*.pth.tar"):
        stage ,epoch ,batch =_extract_metadata (ckpt )
        if stage not in order :
            continue 
        key =(order .index (stage ),epoch ,batch )
        if key >best_key :
            best_key =key 
            best =(ckpt ,stage )
    return best 


CHUNK =32 *1024 *1024 


def _is_gdrive_url (url :str )->bool :
    """Return True if *url* points to Google Drive."""
    import urllib .parse as ul 

    host =ul .urlparse (url ).hostname or ""
    return "drive.google.com"in host or "docs.google.com"in host 


def _is_huggingface_url (url :str )->bool :
    """Return True if *url* points to Hugging Face."""
    import urllib .parse as ul 
    return "huggingface.co"in (ul .urlparse (url ).hostname or "")


def _generic_download (url :str ,dst :Path ,cookie_file :Optional [str ]=None )->None :
    """Download *url* to *dst* with Range resume support."""
    import requests 
    from tqdm import tqdm 

    sess =requests .Session ()

    sess .headers .update ({
    "User-Agent":(
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/125.0 Safari/537.36"
    )
    })


    if _is_huggingface_url (url ):
        token =os .getenv ("HF_TOKEN")or os .getenv ("HUGGINGFACE_HUB_TOKEN")
        if token :
            sess .headers ["Authorization"]=f"Bearer {token }"
    if cookie_file :
        from http .cookiejar import MozillaCookieJar 

        jar =MozillaCookieJar (cookie_file )
        try :
            jar .load ()
        except FileNotFoundError :
            raise FileNotFoundError (f"Cookie file {cookie_file } not found")
        sess .cookies .update (jar )

    resume_at =dst .stat ().st_size if dst .exists ()else 0 
    headers ={"Range":f"bytes={resume_at }-"}if resume_at else {}
    mode ="ab"if resume_at else "wb"
    r =sess .get (url ,stream =True ,headers =headers )
    r .raise_for_status ()
    total =int (r .headers .get ("Content-Length",0 ))+resume_at 
    if "Content-Range"in r .headers :
        try :
            total =int (r .headers ["Content-Range"].split ("/")[-1 ])
        except Exception :
            pass 
        total +=resume_at if resume_at else 0 
    with open (dst ,mode )as f ,tqdm (initial =resume_at ,total =total ,unit ="B",unit_scale =True )as bar :
        for chunk in r .iter_content (CHUNK ):
            if chunk :
                f .write (chunk )
                bar .update (len (chunk ))

    if _is_small_or_html (dst ):
        dst .unlink (missing_ok =True )
        raise RuntimeError (f"Download from {url } did not produce a valid file")


def _gdrive_download (
file_id :str ,
dst :Path ,
expected_sha256 :Optional [str ]=None ,
cookie_file :Optional [str ]=None ,
download_url :Optional [str ]=None ,
)->Path :
    """Stream a Google-Drive file with resume support."""
    import hashlib 
    import re 
    import requests 
    from tqdm import tqdm 

    URL =download_url or "https://docs.google.com/uc?export=download&id="+file_id 
    sess =requests .Session ()
    sess .headers .update ({
    "User-Agent":(
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/125.0 Safari/537.36"
    )
    })
    if cookie_file :
        from http .cookiejar import MozillaCookieJar 

        jar =MozillaCookieJar (cookie_file )
        try :
            jar .load ()
        except FileNotFoundError :
            raise FileNotFoundError (f"Cookie file {cookie_file } not found")
        sess .cookies .update (jar )

    resume_at =dst .stat ().st_size if dst .exists ()else 0 
    params ={"id":file_id ,"confirm":1 }
    response =sess .get (URL ,params =params ,stream =True )
    response .raise_for_status ()

    token =None 
    for k ,v in response .cookies .items ():
        if k .startswith ("download_warning"):
            token =v 
            break 
    if token :
        params ["confirm"]=token 
        response =sess .get (URL ,params =params ,stream =True )
        response .raise_for_status ()

    total =int (response .headers .get ("Content-Length",0 ))+resume_at 

    mode ="ab"if resume_at else "wb"
    with open (dst ,mode )as f ,tqdm (initial =resume_at ,total =total ,unit ="B",unit_scale =True )as bar :
        for chunk in response .iter_content (CHUNK_SIZE ):
            if chunk :
                f .write (chunk )
                bar .update (len (chunk ))

    if expected_sha256 :
        sha =hashlib .sha256 ()
        with open (dst ,"rb")as f :
            for chunk in iter (lambda :f .read (8192 ),b""):
                sha .update (chunk )
        if sha .hexdigest ()!=expected_sha256 :
            dst .unlink (missing_ok =True )
            raise RuntimeError ("Downloaded file failed SHA-256 check")

    if _is_small_or_html (dst ):
        dst .unlink (missing_ok =True )
        raise RuntimeError (f"Download from {URL } did not produce a valid file")

    return dst 


def _download_gdrive_or_fail (url :str ,dst :Path ,expected_sha256 :Optional [str ]=None ,cookie_file :Optional [str ]=None )->None :
    import re 

    m =re .search (r"id=([^&]+)",url )
    if not m :
        m =re .search (r"/d/([^/]+)/",url )
    if not m :
        raise ValueError (f"Could not parse Google Drive ID from {url }")
    file_id =m .group (1 )
    _gdrive_download (file_id ,dst ,expected_sha256 ,cookie_file =cookie_file ,download_url =url )


def _is_small_or_html (path :Path ,min_bytes :int =256 *1024 )->bool :
    if not path .exists ():
        return True 
    size =path .stat ().st_size 
    if size <min_bytes :
        return True 
    with open (path ,"rb")as f :
        head =f .read (512 ).lstrip ()
        if head .startswith (b"<"):
            return True 
    return False 


def _huggingface_download (url :str ,dst :Path ,cookie_file :Optional [str ]=None )->None :
    """Download from Hugging Face with resume support."""
    print (f"Downloading from Hugging Face: {url }")
    _generic_download (url ,dst ,cookie_file )



_RESUME_EPOCH =0 
_RESUME_BATCH =0 
_RESUME_STAGE =None 
_RESUME_OPT =None 
_RESUME_SCHED =None 
_RESUME_SCALER =None 
_RESUME_GLOBAL_STEP =0 


def main ()->None :
    parser =argparse .ArgumentParser (description ="Resume MGM training")
    parser .add_argument ("checkpoint",help ="Checkpoint path or directory")
    parser .add_argument ("--config",required =True ,help ="Path to optim_run_cfg.json")
    parser .add_argument ("--stage",default =None ,help ="Stage to resume")
    parser .add_argument ("--checkpoint-url",default =None ,help ="URL to download checkpoint if missing")
    parser .add_argument ("--cookie-file",default =None ,help ="Cookie file for authenticated downloads")
    args =parser .parse_args ()

    stage_order =[]

    config_path =Path (args .config )
    if not config_path .exists ():
        raise FileNotFoundError (
        f"Config file {config_path } does not exist.\n"
        "Generate it with run_training.py or provide the correct path."
        )
    with open (config_path )as f :
        config =json .load (f )

    if "model"not in config :
        raise KeyError (
        "'model' section missing in config. Did you run run_training.py to create optim_run_cfg.json?"
        )

    stage_order =_stage_order (config )

    ckpt_path =Path (args .checkpoint )
    selected_stage =args .stage 
    download_thread =None 

    if ckpt_path .is_dir ():
        latest =_find_latest_checkpoint (ckpt_path ,stage_order )
        if latest :
            ckpt_path ,selected_stage =latest 
        elif args .checkpoint_url :
            def _download ():
                target =ckpt_path /"downloaded_checkpoint.pth.tar"
                if _is_gdrive_url (args .checkpoint_url ):
                    _download_gdrive_or_fail (args .checkpoint_url ,target ,cookie_file =args .cookie_file )
                elif _is_huggingface_url (args .checkpoint_url ):
                    _huggingface_download (args .checkpoint_url ,target ,cookie_file =args .cookie_file )
                else :
                    _generic_download (args .checkpoint_url ,target ,cookie_file =args .cookie_file )

                if _is_small_or_html (target ):
                    raise RuntimeError ("Download failed – the downloaded file is not a valid checkpoint.")

            download_thread =threading .Thread (target =_download ,daemon =True )
            download_thread .start ()
            ckpt_path =ckpt_path /"downloaded_checkpoint.pth.tar"
        else :
            raise FileNotFoundError (f"No checkpoints in {ckpt_path }")
    elif not ckpt_path .exists ()and args .checkpoint_url :
        def _download ():
            if _is_gdrive_url (args .checkpoint_url ):
                _download_gdrive_or_fail (args .checkpoint_url ,ckpt_path ,cookie_file =args .cookie_file )
            elif _is_huggingface_url (args .checkpoint_url ):
                _huggingface_download (args .checkpoint_url ,ckpt_path ,cookie_file =args .cookie_file )
            else :
                _generic_download (args .checkpoint_url ,ckpt_path ,cookie_file =args .cookie_file )

            if _is_small_or_html (ckpt_path ):
                raise RuntimeError ("Download failed – the downloaded file is not a valid checkpoint.")

        download_thread =threading .Thread (target =_download ,daemon =True )
        download_thread .start ()


    if ckpt_path .exists ():
        try :
            meta_stage ,_ ,_ =_extract_metadata (ckpt_path )
            if meta_stage in stage_order :
                logging .info (f"✅ Checkpoint contains stage metadata: '{meta_stage }'")
                selected_stage =meta_stage 
            else :
                logging .warning (
                f"⚠️ Checkpoint does not contain valid stage metadata. Falling back to --stage argument: {selected_stage }"
                )
        except Exception as e :
            logging .error (f"Could not extract metadata from checkpoint: {e }")

    model =trainer .MixtureOfGeometricExperts (config )


    patch_trainer_for_resume (trainer )
    trainer .validate_epoch =lambda *a ,**k :0.0 


    original_validate =trainer .validate_model_checkpoint_compatibility 

    def _validate_and_set (model_ ,ckpt_path_ ,cfg_ ):
        import torch 
        import re 

        target =ckpt_path_ if ckpt_path_ else ckpt_path 
        while not Path (target ).exists ():
            time .sleep (5 )

        if download_thread :
            download_thread .join ()

        if not Path (target ).exists ():
            raise FileNotFoundError (f"Checkpoint {target } could not be downloaded")

        if _is_small_or_html (Path (target )):
            raise RuntimeError (
            f"Downloaded {Path (target ).name } is not a valid checkpoint "
            f"(size = {Path (target ).stat ().st_size } bytes). "
            "The URL may require authentication or the host returned an HTML page."
            )

        global _RESUME_EPOCH ,_RESUME_BATCH ,_RESUME_STAGE ,_RESUME_OPT 
        global _RESUME_SCHED ,_RESUME_SCALER ,_RESUME_GLOBAL_STEP 


        _RESUME_EPOCH ,_RESUME_BATCH ,_ =load_and_validate_checkpoint (
        model_ ,str (target ),cfg_ ,selected_stage 
        )


        logging .info (f"⚙️ Inferring resume state from filename: {Path (target ).name }")
        filename_match =re .search (r"epoch_(\d+)_batch_(\d+)",Path (target ).name )
        if filename_match :
            fn_epoch =int (filename_match .group (1 ))
            fn_batch =int (filename_match .group (2 ))
            logging .info (
            f"✅ Inferred from filename: epoch={fn_epoch }, batch={fn_batch }. Overriding checkpoint metadata."
            )
            _RESUME_EPOCH =fn_epoch 
            _RESUME_BATCH =fn_batch 
        else :
            logging .warning (
            "⚠️ Could not infer epoch/batch from filename. Relying on internal checkpoint metadata."
            )

        _RESUME_STAGE =selected_stage 
        if _RESUME_STAGE =="curvature_calibration"and _RESUME_EPOCH ==0 and _RESUME_BATCH >50 :
            logging .info ("⏭️  Skipping curvature_calibration based on batch count")
            _RESUME_STAGE ="reasoning_warmup"

        try :
            ckpt_obj =torch .load (str (target ),map_location ="cpu",weights_only =False )
            _RESUME_OPT =ckpt_obj .get ("optimizer")
            _RESUME_SCHED =ckpt_obj .get ("scheduler")
            _RESUME_SCALER =ckpt_obj .get ("scaler")
            _RESUME_GLOBAL_STEP =ckpt_obj .get ("global_step",0 )
            if ckpt_obj .get ("rng_state")is not None :
                torch .set_rng_state (ckpt_obj ["rng_state"])
            if ckpt_obj .get ("numpy_state")is not None :
                import numpy as np 

                np .random .set_state (ckpt_obj ["numpy_state"])
            if ckpt_obj .get ("python_state")is not None :
                import random 

                random .setstate (ckpt_obj ["python_state"])
        except Exception as e :
            print (f"Could not load optimizer/scheduler state: {e }")

        os .environ ["MGM_RESUME_STAGE"]=_RESUME_STAGE or ""
        os .environ ["MGM_RESUME_EPOCH"]=str (_RESUME_EPOCH )
        os .environ ["MGM_RESUME_BATCH"]=str (_RESUME_BATCH )
        os .environ ["MGM_RESUME_GLOBAL_STEP"]=str (_RESUME_GLOBAL_STEP )

        return original_validate (model_ ,ckpt_path_ ,cfg_ )

    trainer .validate_model_checkpoint_compatibility =_validate_and_set 

    def _patched_train_epoch (*args ,**kwargs ):
        import inspect 
        sig =inspect .signature (original_train_epoch )
        epoch_param =next ((name for name ,param in sig .parameters .items ()if name =="epoch"),None )
        if epoch_param :
            epoch_index =list (sig .parameters .keys ()).index (epoch_param )
            epoch =args [epoch_index ]if len (args )>epoch_index else kwargs .get ("epoch",0 )
        else :
            epoch =kwargs .get ("epoch",0 )
        resume_epoch =int (os .environ .get ("MGM_RESUME_EPOCH",-1 ))
        start_batch =int (os .environ .get ("MGM_RESUME_BATCH",0 ))if epoch ==resume_epoch else 0 
        kwargs .pop ("start_batch",None )

        try :
            result =train_epoch_with_resume (
            original_train_epoch ,
            *args ,
            start_batch =start_batch ,
            **kwargs ,
            )
        finally :
            if epoch ==resume_epoch and start_batch >0 :

                os .environ ["MGM_RESUME_BATCH"]="0"
                os .environ ["MGM_RESUME_EPOCH"]=str (-1 )
        return result 

    trainer .train_epoch =_patched_train_epoch 


    original_create_optimizer =getattr (trainer ,"create_geometric_optimizer",None )

    if original_create_optimizer is not None :
        def _patched_create_optimizer (model_ ,stage_cfg_ ,stage_name =None ):
            opt =original_create_optimizer (model_ ,stage_cfg_ ,stage_name )
            if _RESUME_OPT is not None and stage_name ==_RESUME_STAGE :
                try :
                    opt .load_state_dict (_RESUME_OPT )
                    logging .info (f"✅ Optimizer state restored for stage '{stage_name }'")
                except Exception as e :
                    print (f"⚠️ Could not restore optimizer state: {e }")
            return opt 

        trainer .create_geometric_optimizer =_patched_create_optimizer 


    try :
        import torch .optim .lr_scheduler as lrs 

        original_scheduler_cls =lrs .CosineAnnealingWarmRestarts 

        class PatchedScheduler (original_scheduler_cls ):
            def __init__ (self ,*a ,stage_name =None ,**kw ):
                super ().__init__ (*a ,**kw )
                if _RESUME_SCHED is not None and stage_name ==_RESUME_STAGE :
                    try :
                        self .load_state_dict (_RESUME_SCHED )
                        logging .info ("✅ Scheduler state restored")
                    except Exception as e :
                        print (f"⚠️ Could not restore scheduler state: {e }")

        lrs .CosineAnnealingWarmRestarts =PatchedScheduler 
    except Exception :
        pass 


    try :
        from torch .cuda .amp import GradScaler 





        trainer .scaler =None 
        if hasattr (trainer ,"SCALER"):
            trainer .SCALER =None 

        class PatchedScaler (GradScaler ):
            def __init__ (self ,*a ,stage_name =None ,**kw ):
                super ().__init__ (*a ,**kw )

                trainer .scaler =self 
                if hasattr (trainer ,"SCALER"):
                    trainer .SCALER =self 
                if _RESUME_SCALER is not None and stage_name ==_RESUME_STAGE :
                    try :
                        self .load_state_dict (_RESUME_SCALER )
                        logging .info ("✅ GradScaler state restored")
                    except Exception as e :
                        print (f"⚠️ Could not restore GradScaler state: {e }")

        sys .modules ['torch.cuda.amp'].GradScaler =PatchedScaler 
    except Exception :
        pass 




    original_orchestrate =getattr (trainer ,"orchestrate_training_stage",None )

    if original_orchestrate is not None :
        def _patched_orchestrate (stage_name ,*a ,**kw ):
            resume_stage =os .environ .get ("MGM_RESUME_STAGE")
            if resume_stage :
                order =[
                "curvature_calibration",
                "reasoning_warmup",
                "branch_pretrain",
                "gate_train",
                "joint_finetune",
                ]
                try :
                    if order .index (stage_name )<order .index (resume_stage ):
                        logging .info (f"⏭️  Skipping already completed stage: {stage_name }")
                        cur_epoch =kw .get ("current_epoch",0 )
                        gstep =kw .get ("global_step",0 )
                        return cur_epoch ,gstep 
                except ValueError :
                    pass 
            return original_orchestrate (stage_name ,*a ,**kw )

        trainer .orchestrate_training_stage =_patched_orchestrate 


    original_save_checkpoint =getattr (trainer ,"save_checkpoint",None )

    if original_save_checkpoint is not None :
        def _patched_save_checkpoint (state ,filename ="checkpoint.pth.tar"):
            state .setdefault ("stage",state .get ("config",{}).get ("_current_stage"))
            state .setdefault ("stage_completed",False )
            original_save_checkpoint (state ,filename )
        trainer .save_checkpoint =_patched_save_checkpoint 


    sys .argv =[
    "train_geometric_model_v2.py",
    "--config",args .config ,
    "--streaming",
    "--end_to_end",
    "--checkpoint",str (ckpt_path ),
    ]

    trainer .main ()
    os ._exit (0 )


if __name__ =="__main__":
    main ()

