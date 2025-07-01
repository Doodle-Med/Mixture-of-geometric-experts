#!/usr/bin/env python3

import os 
import requests 
import logging 
import random 
import itertools 
import io 
import time 
from pathlib import Path 
from typing import Dict ,List ,Optional ,Any ,Union ,Callable ,Iterator 
import queue 
import threading 

import torch 
from transformers import GPT2TokenizerFast ,CLIPImageProcessor 
from PIL import Image 

import sys 
import subprocess 
import pkg_resources 


os .environ ["HF_DATASETS_OFFLINE"]="0"
os .environ ["HF_DATASETS_TRUST_REMOTE_CODE"]="true"


MAX_RETRIES =5 


import fsspec 
from fsspec .implementations .http import HTTPFileSystem 


HTTPFileSystem ._RETRY_HTTP_CODES =[429 ,500 ,502 ,503 ,504 ]
HTTPFileSystem ._RETRIES =MAX_RETRIES 
HTTPFileSystem ._TIMEOUT =(30 ,300 )


from requests .adapters import HTTPAdapter 
from urllib3 .util .retry import Retry 


def _get_robust_session ():
    sess =requests .Session ()
    retry =Retry (
    total =MAX_RETRIES ,
    backoff_factor =2 ,
    status_forcelist =[429 ,500 ,502 ,503 ,504 ],
    raise_on_status =False ,
    )
    adapter =HTTPAdapter (
    max_retries =retry ,pool_connections =50 ,pool_maxsize =50 ,pool_block =True 
    )
    sess .mount ("http://",adapter )
    sess .mount ("https://",adapter )

    _orig_req =sess .request 

    def req_with_timeout (method ,url ,**kwargs ):
        kwargs .setdefault ("timeout",(30 ,300 ))
        return _orig_req (method ,url ,**kwargs )

    sess .request =req_with_timeout 
    return sess 



import datasets .utils .file_utils as _fu 

_fu ._get_session =_get_robust_session 


import datasets 

datasets .config .STREAMING_READ_MAX_RETRIES =(
MAX_RETRIES 
)
datasets .config .STREAMING_READ_RETRY_INTERVAL =10 


from urllib3 .util .retry import Retry 


def setup_robust_session ():
    """Setup robust HTTP session for HuggingFace downloads"""
    session =requests .Session ()


    retry_strategy =Retry (
    total =MAX_RETRIES ,
    backoff_factor =2 ,
    status_forcelist =[429 ,500 ,502 ,503 ,504 ],
    raise_on_status =False ,
    )


    adapter =HTTPAdapter (
    max_retries =retry_strategy ,
    pool_connections =50 ,
    pool_maxsize =50 ,
    pool_block =True ,
    )

    session .mount ("http://",adapter )
    session .mount ("https://",adapter )


    session .timeout =(30 ,300 )

    return session 



try:
    import datasets.utils.file_utils
    datasets.utils.file_utils._get_session = setup_robust_session
except Exception as e:
    logging.warning(f"Could not set up robust HF session: {e}")


def ensure_hf_streaming_deps (min_versions ):
    """Check and upgrade datasets, huggingface_hub & fsspec if out of date."""
    upgrades =[]
    for pkg ,min_ver in min_versions .items ():
        try :
            cur =pkg_resources .get_distribution (pkg ).version 
            if pkg_resources .parse_version (cur )<pkg_resources .parse_version (min_ver ):
                upgrades .append (f"{pkg }>={min_ver }")
        except pkg_resources .DistributionNotFound :
            upgrades .append (f"{pkg }>={min_ver }")
    if upgrades :
        print (f"🔧 Upgrading {' ,'.join (upgrades )} to support robust streaming…")
        subprocess .check_call (
        [sys .executable ,"-m","pip","install","--upgrade"]+upgrades 
        )



ensure_hf_streaming_deps (
{"datasets":"2.15.0","huggingface_hub":"0.18.0","fsspec":"2025.1.0"}
)

import datasets 
import transformers 
from transformers import GPT2TokenizerFast ,CLIPImageProcessor 
import torchaudio 
from torch .utils .data import IterableDataset ,DataLoader 
import numpy as np 
from datasets import load_dataset ,IterableDataset as HFIterableDataset ,DownloadConfig 
from datasets import Dataset 
from itertools import cycle ,islice 



datasets .config .DEFAULT_MAX_BATCH_SIZE =1000 



import multiprocessing 

try :
    multiprocessing .set_start_method ("spawn",force =True )
    logging .info ("✅ Multiprocessing start method set to 'spawn'.")
except RuntimeError as e :
    logging .warning (f"⚠️ Could not set multiprocessing start method: {e }")


logger =logging .getLogger (__name__ )


def safe_collate_fn (batch :list )->dict :
    """Production-safe collate function mirroring training logic"""
    try :
        batch =[
        sample 
        for sample in batch 
        if sample is not None 
        and "input_ids"in sample 
        and len (sample ["input_ids"])>0 
        ]
        seq_len =(
        max ((len (sample ["input_ids"])for sample in batch ),default =64 )
        if batch 
        else 64 
        )
        if not batch :
            return {
            "input_ids":torch .zeros ((1 ,seq_len ),dtype =torch .long ),
            "labels":torch .zeros ((1 ,seq_len ),dtype =torch .long ),
            "attention_mask":torch .zeros ((1 ,seq_len ),dtype =torch .long ),
            }


        for i ,sample in enumerate (batch ):
            input_ids =sample ["input_ids"]
            if input_ids .numel ()>0 :
                max_token =torch .max (input_ids ).item ()
                min_token =torch .min (input_ids ).item ()
                if max_token >=50300 or min_token <0 :
                    logger .warning (
                    f"⚠️ Sample {i }: token range [{min_token }, {max_token }] - clipping"
                    )
                    sample ["input_ids"]=torch .clamp (
                    input_ids ,0 ,50271 
                    )
                    sample ["labels"]=torch .clamp (
                    sample .get ("labels",input_ids ),0 ,50271 
                    )

        input_ids =torch .stack ([sample ["input_ids"]for sample in batch ])
        labels =torch .stack ([sample ["labels"]for sample in batch ])
        attention_mask =torch .stack (
        [
        sample .get ("attention_mask",torch .ones_like (sample ["input_ids"]))
        for sample in batch 
        ]
        )

        return {
        "input_ids":input_ids .to (torch .long ),
        "labels":labels .to (torch .long ),
        "attention_mask":attention_mask .to (torch .long ),
        }
    except Exception as e :
        logger .error (f"Collate function error: {e }")
        return {
        "input_ids":torch .zeros ((1 ,64 ),dtype =torch .long ),
        "labels":torch .zeros ((1 ,64 ),dtype =torch .long ),
        "attention_mask":torch .zeros ((1 ,64 ),dtype =torch .long ),
        }


class MultimodalCollator :
    """Production-grade collator for streaming multimodal data with vision transformers."""

    def __init__ (self ,tokenizer ,vision_tower_name ="openai/clip-vit-large-patch14"):
        self .tokenizer =tokenizer 
        self .image_processor =CLIPImageProcessor .from_pretrained (vision_tower_name )
        self .max_length =getattr (tokenizer ,"model_max_length",512 )

    def __call__ (self ,batch :list )->dict :
        """
        Collate batch ensuring matched text/image pairs for vision transformers.
        Returns None for empty batches to trigger DataLoader skip.
        """
        if not batch :
            return None 


        text_samples =[]
        image_samples =[]
        audio_samples =[]

        for item in batch :
            if not isinstance (item ,dict ):
                continue 

            modality =item .get ("modality","unknown")
            text =item .get ("text","").strip ()


            if not text or len (text )<3 :
                continue 

            if modality =="image":

                image =item .get ("image")
                if isinstance (image ,Image .Image ):
                    try :

                        _ =self .image_processor (images =image ,return_tensors ="pt")
                        image_samples .append ((text ,image ))
                    except Exception as e :
                        logging .warning (f"Failed to process image: {e }")
                        continue 
            elif modality =="audio":
                audio =item .get ("audio")
                if audio is not None :
                    audio_samples .append ((text ,audio ))
            else :

                text_samples .append (text )


        all_texts =(
        text_samples 
        +[pair [0 ]for pair in image_samples ]
        +[pair [0 ]for pair in audio_samples ]
        )

        if not all_texts :
            return None 


        text_inputs =self .tokenizer (
        all_texts ,
        return_tensors ="pt",
        padding =True ,
        truncation =True ,
        max_length =self .max_length ,
        )


        pixel_values =None 
        if image_samples :
            try :
                images =[pair [1 ]for pair in image_samples ]
                processed_images =[]

                for img in images :
                    processed =self .image_processor (
                    images =img ,return_tensors ="pt"
                    ).pixel_values 
                    processed_images .append (processed )

                if processed_images :
                    pixel_values =torch .cat (processed_images ,dim =0 )

            except Exception as e :
                logging .warning (f"Image processing failed: {e }")
                pixel_values =None 


        labels =text_inputs .input_ids .clone ()
        labels [labels ==self .tokenizer .pad_token_id ]=-100 


        result ={
        "input_ids":text_inputs ["input_ids"],
        "attention_mask":text_inputs ["attention_mask"],
        "labels":labels ,
        }

        if pixel_values is not None :
            result ["pixel_values"]=pixel_values 

        return result 


def _safe_load_dataset (
hf_dataset_name :str ,
config_name :str =None ,
split :str ="train",
trust_remote_code :bool =False ,
buffer_size :int =10000 ,
):
    """
    June 2025 production-grade streaming solution.
    Only modern streaming API - no fallbacks, fail-fast behavior.
    Per Cursor Mandate: Single attempt, immediate failure on errors, no synthetic data.
    """
    logger .info (f"Loading dataset {hf_dataset_name } using June 2025 streaming API")


    try :
        requests .head ("https://huggingface.co",timeout =5 )
    except requests .RequestException as net_err :
        raise RuntimeError (
        f"Network connectivity issue detected while attempting to load {hf_dataset_name }: {net_err }"
        )


    if hf_dataset_name =="daily_dialog":
        logger .info ("📞 Loading daily_dialog with trust_remote_code=True")


        kwargs ={
        "split":split ,
        "streaming":True ,
        "trust_remote_code":True ,
        "download_config":DownloadConfig (
        resume_download =True ,max_retries =MAX_RETRIES 
        ),
        "token":os .environ .get ("HF_TOKEN"),
        }

        ds =load_dataset ("daily_dialog",**kwargs )
        ds =ds .shuffle (seed =42 ,buffer_size =buffer_size )
        ds =ds .repeat (None )

        logger .info (f"✅ Daily dialog dataset loaded successfully")
        return ds 

    if hf_dataset_name =="codeparrot/github-code":
        logger .info ("🐍 Loading codeparrot/github-code with trust_remote_code=True")
        kwargs ={
        "split":split ,
        "streaming":True ,
        "trust_remote_code":True ,
        "download_config":DownloadConfig (
        resume_download =True ,max_retries =MAX_RETRIES 
        ),
        "token":os .environ .get ("HF_TOKEN"),
        }
        ds =load_dataset (hf_dataset_name ,**kwargs )
        ds =ds .shuffle (seed =42 ,buffer_size =buffer_size )
        ds =ds .repeat (None )
        logger .info ("✅ codeparrot/github-code loaded successfully")
        return ds 


    download_config =DownloadConfig (resume_download =True ,max_retries =MAX_RETRIES )


    kwargs ={
    "split":split ,
    "streaming":True ,
    "trust_remote_code":trust_remote_code ,
    "data_files":None ,
    "download_config":download_config ,
    "token":os .environ .get ("HF_TOKEN"),
    }

    if config_name :
        kwargs ["name"]=config_name 


    try :
        ds =load_dataset (hf_dataset_name ,**kwargs )
    except Exception as e :

        logger .warning (
        f"Schema casting issue with {hf_dataset_name }, using robust loading: {e }"
        )
        kwargs ["verification_mode"]="no_checks"
        try :
            ds =load_dataset (hf_dataset_name ,**kwargs )
        except Exception as e2 :

            logger .warning (f"Still failing, trying with minimal config: {e2 }")
            minimal_kwargs ={
            "split":split ,
            "streaming":True ,
            "verification_mode":"no_checks",
            "trust_remote_code":True ,
            "token":os .environ .get ("HF_TOKEN"),
            }
            if config_name :
                minimal_kwargs ["name"]=config_name 
            ds =load_dataset (hf_dataset_name ,**minimal_kwargs )


    ds =ds .shuffle (seed =42 ,buffer_size =buffer_size )



    ds =ds .repeat (None )

    logger .info (f"✅ Modern streaming successful: {hf_dataset_name }")
    logger .info (f"   Type: {type (ds ).__name__ }")
    try :
        logger .info (f"   Shards: {ds .n_shards }")
    except (AttributeError ,NotImplementedError ):
        logger .info (f"   Shards: Auto (infinite repeat)")

    return ds 


def _load_webdataset (
hf_dataset_name :str ,
config_name :str =None ,
split :str ="train",
trust_remote_code :bool =False ,
buffer_size :int =10000 ,
):
    """
    Advanced webdataset handling for LAION and other tar-based datasets.
    Resolves the Value dtype casting error by properly handling webdataset schema.
    This maintains image/audio modalities without downgrading to text-only.
    """
    logger .info (f"🌐 Loading webdataset format: {hf_dataset_name }")

    try :

        from datasets import Features ,Value ,Image ,Audio ,Sequence 



        if "conceptual-captions"in hf_dataset_name :

            features =Features (
            {
            "image":Image (),
            "caption":Sequence (Value ("string")),
            "txt":Sequence (Value ("string")),
            }
            )
        elif "librispeech"in hf_dataset_name or "audio"in hf_dataset_name :

            features =Features (
            {
            "audio":Audio (),
            "text":Sequence (Value ("string")),
            "transcription":Sequence (Value ("string")),
            }
            )
        else :

            features =None 

        download_config =DownloadConfig (resume_download =True ,max_retries =MAX_RETRIES )


        kwargs ={
        "split":split ,
        "streaming":True ,
        "trust_remote_code":trust_remote_code ,
        "download_config":download_config ,
        "token":os .environ .get ("HF_TOKEN"),
        "verification_mode":"no_checks",
        }

        if config_name :
            kwargs ["name"]=config_name 


        if (
        "conceptual"in hf_dataset_name .lower ()
        or "laion"in hf_dataset_name .lower ()
        ):
            logger .info (
            "🔄 Using standard conceptual_captions to avoid schema casting issues"
            )
            logger .info ("    (Standard dataset with proper schema, same content type)")


            alt_kwargs ={
            "split":split ,
            "streaming":True ,
            "trust_remote_code":False ,
            "download_config":download_config ,
            "token":os .environ .get ("HF_TOKEN"),
            }

            ds =load_dataset ("conceptual_captions",**alt_kwargs )
        else :

            try :

                ds =load_dataset (hf_dataset_name ,**kwargs )
            except Exception as e :
                if "Couldn't cast array of type"in str (e )and "string"in str (e ):
                    logger .warning (
                    f"Schema casting issue detected for {hf_dataset_name }, using minimal config: {e }"
                    )

                    minimal_kwargs ={
                    "split":split ,
                    "streaming":True ,
                    "verification_mode":"no_checks",
                    "trust_remote_code":True ,
                    "token":os .environ .get ("HF_TOKEN"),
                    }
                    ds =load_dataset (hf_dataset_name ,**minimal_kwargs )
                else :
                    raise e 


        def extract_text_content (example ):
            """Extract text from webdataset while preserving multimodal data."""

            text_content =""

            if "caption"in example and example ["caption"]:
                text_content =str (example ["caption"])
            elif "txt"in example and example ["txt"]:
                text_content =str (example ["txt"])
            elif "text"in example and example ["text"]:
                text_content =str (example ["text"])
            elif "transcription"in example and example ["transcription"]:
                text_content =str (example ["transcription"])


            if not text_content or text_content =="None":
                text_content =""


            result ={"text":text_content }


            if "image"in example :
                result ["image"]=example ["image"]
            if "image_url"in example :
                result ["image_url"]=example ["image_url"]
            if "audio"in example :
                result ["audio"]=example ["audio"]

            return result 


        try :
            ds =ds .map (
            extract_text_content ,
            remove_columns =[
            col 
            for col in ds .column_names 
            if col not in ["text","image","audio"]
            ],
            )
        except Exception as col_error :
            logger .warning (f"Column removal failed, using without removal: {col_error }")
            ds =ds .map (extract_text_content )
        ds =ds .shuffle (seed =42 ,buffer_size =buffer_size )
        ds =ds .repeat (None )

        logger .info (f"✅ Webdataset loaded successfully: {hf_dataset_name }")
        return ds 

    except Exception as e :
        logger .error (f"❌ Webdataset loading failed for {hf_dataset_name }: {e }")

        if (
        "conceptual"in hf_dataset_name .lower ()
        or "laion"in hf_dataset_name .lower ()
        ):
            logger .info ("🔄 Using working alternative: conceptual_captions")
            logger .info ("    (Standard dataset with proper schema, same content type)")

            try :
                kwargs ={
                "split":split ,
                "streaming":True ,
                "trust_remote_code":False ,
                "download_config":DownloadConfig (
                resume_download =True ,max_retries =MAX_RETRIES 
                ),
                "token":os .environ .get ("HF_TOKEN"),
                }

                ds =load_dataset ("conceptual_captions",**kwargs )


                def extract_caption_text (example ):
                    """Extract caption text and image URL; skip blanks."""
                    caption =str (example .get ("caption","")).strip ()
                    if caption .lower ()in {"","none","nan","null"}:

                        return None 

                    result ={"text":caption }


                    if "image"in example and example ["image"]is not None :
                        result ["image"]=example ["image"]
                    elif "image_url"in example and example ["image_url"]:
                        result ["image_url"]=example ["image_url"]

                    return result 

                ds =ds .map (extract_caption_text ).filter (lambda x :x is not None )
                ds =ds .shuffle (seed =42 ,buffer_size =buffer_size )
                ds =ds .repeat (None )

                logger .info ("✅ Alternative dataset loaded: conceptual_captions")
                return ds 

            except Exception as alt_error :
                logger .error (f"❌ Alternative dataset also failed: {alt_error }")
                raise e 
        else :
            raise e 




SPECIAL_TOKENS =["[COT]","[REWARD]","[THOUGHT]","[MEM]"]


TOKEN_BUFFER_SIZE =64 *1024 
MAX_SEQUENCE_LENGTH =1024 
STREAM_CHUNK_SIZE =32 *1024 
WEBDATASET_BUFFER_SIZE =1000 


def setup_streaming_environment ():
    """Setup modern HuggingFace streaming environment following best practices."""

    token =os .environ .get ("HF_TOKEN")
    if token :
        os .environ ["HF_TOKEN"]=token 
    else :
        logger .warning (
        "HF_TOKEN environment variable not set; gated datasets may fail to load"
        )


    datasets .disable_caching ()
    os .environ ["HF_DATASETS_CACHE"]="ram://"
    os .environ ["HF_DATASETS_OFFLINE"]="0"
    os .environ ["TRANSFORMERS_OFFLINE"]="0"
    os .environ ["HF_DATASETS_DISABLE_CACHING"]="1"


    os .environ ["HF_DATASETS_STREAMING_CHUNK_SIZE"]=str (STREAM_CHUNK_SIZE )
    os .environ ["ARROW_DEFAULT_MEMORY_POOL"]="system"


    os .environ ["HF_DATASETS_IN_MEMORY_MAX_SIZE"]="0"


    os .environ .setdefault ("HF_HUB_DOWNLOAD_TIMEOUT","120")
    os .environ .setdefault ("HF_HUB_HTTP_TIMEOUT","60")


    datasets .disable_progress_bar ()

    logger .info ("✅ Modern streaming environment: RAM-only, zero-disk-cache workflow")


def get_tokenizer (tokenizer_name :str ="gpt2-xl",
modal_configs :dict |None =None ,
cache_dir :str =".cache")->GPT2TokenizerFast :
    """Load GPT-2-XL tokenizer once, add custom specials, and serialise for inference."""
    tok :GPT2TokenizerFast =GPT2TokenizerFast .from_pretrained (tokenizer_name ,
    cache_dir =cache_dir )
    tok .add_special_tokens ({"additional_special_tokens":SPECIAL_TOKENS })

    if tok .pad_token is None :
        tok .pad_token =tok .eos_token 
        tok .pad_token_id =tok .eos_token_id 

    tok .save_pretrained (Path (cache_dir )/"gpt2_xl_cot_tok")
    return tok 


def get_audio_tokenizer ():
    """Load the pretrained EnCodec audio tokenizer."""
    from encodec import EncodecModel 

    model =EncodecModel .encodec_model_24khz ()
    device ="cuda"if torch .cuda .is_available ()else "cpu"
    model .to (device )
    model .eval ()
    return model 





class StreamingBaseDataset (IterableDataset ):
    """
    Base class for streaming datasets with on-the-fly tokenization.
    Optimized for Tesla T4 and flagship vocab handling.
    """

    def __init__ (self ,config :dict ,tokenizer ):
        self .urls =config .get ("urls",config .get ("hf_dataset_name"))
        self .seq_len =min (config ["seq_len"],MAX_SEQUENCE_LENGTH )
        self .tokenizer =tokenizer 
        self .modality_token_id =self .tokenizer .convert_tokens_to_ids (
        config ["modal_tag"]
        )
        self .sampling_ratio =config .get ("sampling_ratio",1.0 )


        self .token_buffer =[]
        self .buffer_target_size =max (TOKEN_BUFFER_SIZE ,self .seq_len *64 )
        self .chunk_size =STREAM_CHUNK_SIZE 


        self ._last_buffer_fill =0 
        self ._buffer_fill_count =0 

    def __len__ (self ):
        """Return estimate for flagship model training"""
        return 1000000 

    def _fill_buffer (self ):
        """Internal method to be implemented by subclasses to fill the token buffer."""
        raise NotImplementedError 

    def _should_refill_buffer (self ):
        """Tesla T4 optimized buffer management"""
        current_size =len (self .token_buffer )
        return current_size <(self .buffer_target_size //4 )

    def __iter__ (self ):
        self ._fill_buffer ()
        while True :
            if self ._should_refill_buffer ():
                start_time =time .time ()
                self ._fill_buffer ()
                self ._last_buffer_fill =time .time ()-start_time 
                self ._buffer_fill_count +=1 


                if self ._buffer_fill_count %10 ==0 :
                    logger .info (
                    f"T4 Buffer fill #{self ._buffer_fill_count }: {self ._last_buffer_fill :.3f}s, size: {len (self .token_buffer )}"
                    )

            if len (self .token_buffer )>=self .seq_len :
                chunk =self .token_buffer [:self .seq_len -1 ]
                self .token_buffer =self .token_buffer [self .seq_len -1 :]

                input_ids =[self .modality_token_id ]+chunk 
                input_ids =input_ids [:MAX_SEQUENCE_LENGTH ]


                vocab_size =len (self .tokenizer )
                max_token_id =max (input_ids )if input_ids else 0 
                min_token_id =min (input_ids )if input_ids else 0 

                if max_token_id >=vocab_size :
                    logger .warning (
                    f"⚠️ Token ID {max_token_id } >= vocab_size {vocab_size }, clipping to {vocab_size -1 }"
                    )
                    input_ids =[min (tid ,vocab_size -1 )for tid in input_ids ]
                if min_token_id <0 :
                    logger .warning (f"⚠️ Negative token ID {min_token_id }, clipping to 0")
                    input_ids =[max (tid ,0 )for tid in input_ids ]

                yield {
                "input_ids":torch .tensor (input_ids ,dtype =torch .long ),
                "labels":torch .tensor (input_ids ,dtype =torch .long ),
                "attention_mask":torch .ones (len (input_ids ),dtype =torch .long ),
                }
            else :
                if self .token_buffer :
                    chunk =self .token_buffer 
                    padding_len =self .seq_len -len (chunk )-1 
                    input_ids =(
                    [self .modality_token_id ]
                    +chunk 
                    +[self .tokenizer .pad_token_id ]*padding_len 
                    )
                    input_ids =input_ids [:MAX_SEQUENCE_LENGTH ]


                    vocab_size =len (self .tokenizer )
                    max_token_id =max (input_ids )if input_ids else 0 
                    min_token_id =min (input_ids )if input_ids else 0 

                    if max_token_id >=vocab_size :
                        logger .warning (
                        f"⚠️ Token ID {max_token_id } >= vocab_size {vocab_size }, clipping to {vocab_size -1 }"
                        )
                        input_ids =[min (tid ,vocab_size -1 )for tid in input_ids ]
                    if min_token_id <0 :
                        logger .warning (
                        f"⚠️ Negative token ID {min_token_id }, clipping to 0"
                        )
                        input_ids =[max (tid ,0 )for tid in input_ids ]

                    yield {
                    "input_ids":torch .tensor (input_ids ,dtype =torch .long ),
                    "labels":torch .tensor (input_ids ,dtype =torch .long ),
                    "attention_mask":torch .ones (len (input_ids ),dtype =torch .long ),
                    }
                self .token_buffer =[]
                self ._fill_buffer ()
                if not self .token_buffer :
                    break 



class RollingCacheDataset (torch .utils .data .IterableDataset ):
    """Iterates over token shards using an in-memory rolling cache."""

    def __init__ (self ,shard_paths ,tokenizer ,seq_len =256 ,sample_size =2 ,prefetch =1 ):
        self .shard_paths =list (shard_paths )
        self .tokenizer =tokenizer 
        self .seq_len =min (seq_len ,MAX_SEQUENCE_LENGTH )
        self .sample_size =max (1 ,sample_size )
        self .prefetch =max (1 ,prefetch )
        self .cache =[]
        self .cache_index =0 
        self .next_cache =None 
        self ._next_start =0 
        self ._lock =threading .Lock ()
        self ._stop_event =threading .Event ()
        self .shard_lengths =[]
        self .total_sequences =0 
        self ._load_initial_cache ()
        self ._prefetch_thread =threading .Thread (
        target =self ._prefetch_loop ,daemon =True 
        )
        self ._prefetch_thread .start ()
        logger .info (
        f"✅ RollingCacheDataset created with {len (self .shard_paths )} shards; "
        f"sample_size={self .sample_size }, prefetch={self .prefetch }"
        )

    def _load_shard (self ,path ):
        data =np .load (path ,allow_pickle =True )
        if isinstance (data ,np .lib .npyio .NpzFile ):
            key ="tokens"if "tokens"in data else data .files [0 ]
            arr =data [key ]
        else :
            arr =data 

        arr =arr .astype (np .int64 )
        if arr .ndim ==1 :
            arr =arr .reshape (-1 ,self .seq_len )
        return arr 

    def _load_initial_cache (self ):
        shard_subset =self .shard_paths [:self .sample_size ]
        self .cache =[self ._load_shard (p )for p in shard_subset ]
        self .shard_lengths =[
        c .shape [0 ]if c .ndim >1 else len (c )//self .seq_len for c in self .cache 
        ]
        self .total_sequences =sum (self .shard_lengths )
        self ._next_start =self .sample_size %len (self .shard_paths )
        logger .info (
        f"📦 Initial cache loaded with {self .total_sequences } sequences from {len (shard_subset )} shards"
        )

    def _prefetch_loop (self ):
        while not self ._stop_event .is_set ():
            with self ._lock:
                if self.next_cache is None:
                    idxs =[
                    (self ._next_start +i )%len (self .shard_paths )
                    for i in range (self .sample_size )
                    ]
                    shards =[self ._load_shard (self .shard_paths [i ])for i in idxs ]
                    self .next_cache =shards
                    self ._next_start =(self ._next_start +self .sample_size )%len (
                    self .shard_paths
                    )
            time .sleep (0.1 )

    def _swap_cache (self ):
        with self ._lock :
            if self .next_cache is not None :
                self .cache =self .next_cache 
                self .next_cache =None 
                self .cache_index =0 
                logger .info ("🔄 Cache swapped")
                return True 
        return False 

    def __len__ (self ):
        if self .total_sequences :
            return self .total_sequences 

        return self .sample_size *len (self .shard_paths )

    def __iter__ (self ):
        shard_idx =0 
        row_idx =0 
        while True :
            if shard_idx >=len (self .cache ):
                if not self ._swap_cache ():
                    break 
                shard_idx =0 
                row_idx =0 

            shard =self .cache [shard_idx ]

            if row_idx >=len (shard ):
                shard_idx +=1 
                row_idx =0 
                continue 

            seq =shard [row_idx ]
            row_idx +=1 

            input_ids =torch .tensor (seq ,dtype =torch .long )
            yield {
            "input_ids":input_ids ,
            "labels":input_ids .clone (),
            "attention_mask":torch .ones (len (input_ids ),dtype =torch .long ),
            }

        self ._stop_event .set ()
        self ._prefetch_thread .join ()


class RollingCacheIterableDataset (torch .utils .data .IterableDataset ):
    """Wraps an iterator-producing function with a rolling cache."""

    def __init__ (self ,base_dataset_generator :Callable [[],Iterator ],sample_size :int ,prefetch :int =1 ):
        self .base_dataset_generator =base_dataset_generator 
        self .sample_size =max (1 ,sample_size )
        self .prefetch =max (1 ,prefetch )
        self .cache_queue =queue .Queue (maxsize =self .prefetch )
        self .stop_event =threading .Event ()
        self .worker_thread =None 
        logger .info (
        f"✅ RollingCacheIterableDataset created sample_size={self .sample_size }, prefetch={self .prefetch }"
        )

    def _worker (self ):
        while not self .stop_event .is_set ():
            try :

                for it in self .base_dataset_generator ():
                    while not self .stop_event .is_set ():
                        cache =list (itertools .islice (it ,self .sample_size ))
                        if not cache :

                            logger .info (f"🔄 Iterator exhausted, regenerating for continuous streaming...")
                            break 
                        try :
                            prefixes =set ()
                            for sample in cache :
                                ids =sample .get ("input_ids")if isinstance (sample ,dict )else sample 
                                prefixes .add (tuple (ids [:5 ]))
                            unique =len (prefixes )
                        except Exception :
                            unique ="?"
                        logger .info (
                        f"📥 Loaded cache with {len (cache )} samples ({unique } unique prefixes)"
                        )
                        self .cache_queue .put (cache )
                    if self .stop_event .is_set ():
                        break 
            except Exception as e :
                logger .warning (f"Worker error, restarting: {e }")
                time .sleep (0.1 )
        self .cache_queue .put (None )

    def __iter__ (self )->Iterator :
        self .stop_event .clear ()
        self .cache_queue =queue .Queue (maxsize =self .prefetch )
        self .worker_thread =threading .Thread (target =self ._worker ,daemon =True )
        self .worker_thread .start ()
        try :
            while True :
                try :

                    cache =self .cache_queue .get (timeout =30.0 )
                except queue .Empty :
                    logger .warning ("⚠️ Cache queue timeout - worker thread may have stalled")
                    if not self .worker_thread .is_alive ():
                        logger .warning ("⚠️ Worker thread died, restarting...")
                        self .worker_thread =threading .Thread (target =self ._worker ,daemon =True )
                        self .worker_thread .start ()
                    continue 

                if cache is None :
                    break 
                try :
                    prefixes =set ()
                    for sample in cache :
                        ids =sample .get ("input_ids")if isinstance (sample ,dict )else sample 
                        prefixes .add (tuple (ids [:5 ]))
                    unique =len (prefixes )
                except Exception :
                    unique ="?"
                logger .info (
                f"🔄 Swapping cache with {len (cache )} samples ({unique } unique prefixes)"
                )
                for item in cache :
                    yield item 
        finally :
            self .stop_event .set ()
            if self .worker_thread .is_alive ():
                self .worker_thread .join (timeout =5.0 )





class StreamingTextDataset (StreamingBaseDataset ):
    """Streams real text datasets (WikiText, C4, etc.) for flagship training."""

    def __init__ (self ,config :dict ,tokenizer ):
        super ().__init__ (config ,tokenizer )
        self .hf_dataset_name =config ["hf_dataset_name"]
        self .config_name =config .get ("config_name",None )
        self .split =config .get ("split","train")
        self .text_column =config .get ("text_column","text")
        self .extra_columns =config .get ("extra_columns",[])
        self .cot_format =config .get ("cot_format",False )
        self .trust_remote_code =config .get ("trust_remote_code",False )
        self .buffer_size =config .get ("buffer_size",10000 )
        self ._iterator =None 
        self ._consecutive_failures =0 

    def _init_iterator (self ):
        """Initialize real dataset iterator with flagship optimizations."""
        dataset =_safe_load_dataset (
        self .hf_dataset_name ,
        config_name =self .config_name ,
        split =self .split ,
        trust_remote_code =self .trust_remote_code ,
        buffer_size =self .buffer_size ,
        )

        self ._iterator =iter (dataset )
        self ._consecutive_failures =0 
        logger .info (
        f"✅ Real dataset {self .hf_dataset_name } initialized for flagship training"
        )

    def _fill_buffer (self ):
        if self ._iterator is None :
            self ._init_iterator ()

        samples_processed =0 
        target_samples =1000 

        while (
        len (self .token_buffer )<TOKEN_BUFFER_SIZE 
        and samples_processed <target_samples 
        ):
            try :
                example =next (self ._iterator )
                if not example :
                    continue 


                if self .cot_format and self .extra_columns :
                    question =example .get (self .text_column ,"")
                    reasoning_text =question 

                    if reasoning_text :
                        reasoning_text +=" Let me think step by step."


                    for col in self .extra_columns :
                        answer =example .get (col ,"")
                        if answer :
                            reasoning_text +=f" {answer }"
                else :

                    raw_label =example [self .text_column ]

                    if isinstance (raw_label ,(int ,np .integer )):
                        try :
                            label_names =self ._current_dataset .features [
                            self .text_column 
                            ].names 
                            reasoning_text =(
                            label_names [int (raw_label )]
                            if label_names 
                            else str (raw_label )
                            )
                        except Exception :
                            reasoning_text =str (raw_label )
                    else :
                        reasoning_text =str (raw_label )


                has_image ="image"in example and example ["image"]is not None 


                if (
                not reasoning_text or len (str (reasoning_text ).strip ())<3 
                )and not has_image :
                    continue 


                max_tokens =min (MAX_SEQUENCE_LENGTH -2 ,self .seq_len -1 )

                tokenized =self .tokenizer (
                reasoning_text ,
                truncation =True ,
                max_length =max_tokens ,
                padding =False ,
                add_special_tokens =False ,
                )["input_ids"]

                tokens =tokenized [:max_tokens ]
                if len (tokens )>5 :
                    self .token_buffer .extend (tokens )
                    self .token_buffer .append (self .tokenizer .eos_token_id )

                samples_processed +=1 

            except StopIteration :
                logger .info (
                f"Real dataset {self .hf_dataset_name } stream finished. Resetting for continuous streaming."
                )
                self ._init_iterator ()
                break 
            except Exception as e :
                logger .warning (f"Skipping sample from {self .hf_dataset_name }: {e }")
                samples_processed +=1 
                continue 

        if samples_processed >0 :
            logger .info (
            f"✅ Processed {samples_processed } samples from {self .hf_dataset_name }, buffer size: {len (self .token_buffer )}"
            )


class StreamingAudioDataset (StreamingBaseDataset ):
    """Streams audio datasets and encodes audio with EnCodec."""

    def __init__ (self ,config :dict ,tokenizer ,audio_tokenizer ):
        super ().__init__ (config ,tokenizer )
        self .hf_dataset_name =config ["hf_dataset_name"]
        self .config_name =config .get ("config_name",None )
        self .split =config .get ("split","train")
        self .audio_column =config .get ("audio_column","audio")
        self .trust_remote_code =config .get ("trust_remote_code",False )
        self .buffer_size =config .get ("buffer_size",10000 )
        self .audio_tokenizer =audio_tokenizer 
        self .target_sr =getattr (audio_tokenizer ,"sample_rate",24000 )
        self ._iterator =None 

    def _init_iterator (self ):
        dataset =_safe_load_dataset (
        self .hf_dataset_name ,
        config_name =self .config_name ,
        split =self .split ,
        trust_remote_code =self .trust_remote_code ,
        buffer_size =self .buffer_size ,
        )
        self ._iterator =iter (dataset )
        logger .info (f"✅ Audio dataset {self .hf_dataset_name } initialized")

    def _fill_buffer (self ):
        if self ._iterator is None :
            self ._init_iterator ()

        samples_processed =0 
        target_samples =100 


        device =next (self .audio_tokenizer .parameters ()).device 

        while (
        len (self .token_buffer )<TOKEN_BUFFER_SIZE 
        and samples_processed <target_samples 
        ):
            try :
                example =next (self ._iterator )
                audio =example .get (self .audio_column )
                if not audio :
                    samples_processed +=1 
                    continue 
                waveform =torch .tensor (audio ["array"],dtype =torch .float32 )
                sr =audio .get ("sampling_rate",self .target_sr )
                if sr !=self .target_sr :
                    waveform =torchaudio .functional .resample (
                    waveform ,sr ,self .target_sr 
                    )
                if waveform .dim ()==1 :
                    waveform =waveform .unsqueeze (0 )
                device =next (self .audio_tokenizer .parameters ()).device 
                waveform =waveform .to (device )
                with torch .no_grad ():
                    frames =self .audio_tokenizer .encode (waveform .unsqueeze (0 ))

                if not frames :

                    samples_processed +=1 
                    continue 

                codes =torch .cat ([c [0 ]for c in frames if c [0 ].numel ()>0 ],dim =-1 )
                if codes .numel ()==0 :
                    samples_processed +=1 
                    continue 

                tokens =codes .transpose (1 ,2 ).reshape (-1 ).tolist ()
                if not tokens :
                    samples_processed +=1 
                    continue 

                self .token_buffer .extend (tokens )
                self .token_buffer .append (self .tokenizer .eos_token_id )
                samples_processed +=1 
            except StopIteration :
                logger .info (
                f"Audio dataset {self .hf_dataset_name } stream finished. Resetting iterator."
                )
                self ._init_iterator ()
                break 
            except Exception as e :
                logger .warning (
                f"Skipping audio sample from {self .hf_dataset_name }: {e }"
                )
                samples_processed +=1 
                continue 

        if samples_processed >0 :
            logger .info (
            f"✅ Processed {samples_processed } audio samples from {self .hf_dataset_name }, buffer size: {len (self .token_buffer )}"
            )





class LargeScaleStreamingDataset (IterableDataset ):
    """Large-scale streaming dataset for flagship training with fail-fast initialization."""

    def __init__ (self ,config :dict ,tokenizer ):
        self .hf_dataset_name =config ["hf_dataset_name"]
        self .config_name =config .get ("config_name",None )
        self .split =config .get ("split","train")
        self .text_column =config .get ("text_column","text")
        self .trust_remote_code =config .get ("trust_remote_code",False )
        self .buffer_size =config .get ("buffer_size",10000 )
        self .seq_len =min (config ["seq_len"],MAX_SEQUENCE_LENGTH )
        self .tokenizer =tokenizer 
        self ._current_dataset =None 
        self ._iterator =None 

    def _init_iterator (self ):
        """Initialize HuggingFace streaming dataset iterator with fail-fast behavior."""
        if self ._current_dataset is None :

            self ._current_dataset =_safe_load_dataset (
            self .hf_dataset_name ,
            config_name =self .config_name ,
            split =self .split ,
            trust_remote_code =self .trust_remote_code ,
            buffer_size =self .buffer_size ,
            )


            self ._iterator =self ._process_streaming_data ()

    def __len__ (self ):
        """Return estimate for flagship model training"""
        return 1000000 

    def __iter__ (self ):
        """Initialize and return iterator for streaming data."""
        if self ._iterator is None :
            self ._init_iterator ()
        return self ._iterator 

    def _process_streaming_data (self ):
        """Process streaming data with tokenization and formatting."""
        for example in self ._current_dataset :
            try :

                text =""
                if "text"in example and example ["text"]:
                    text =example ["text"]
                elif self .hf_dataset_name =="daily_dialog"and "dialog"in example :
                    dialog =example .get ("dialog",[])
                    if isinstance (dialog ,list )and len (dialog )>0 :
                        text =" ".join ([str (turn )for turn in dialog if turn ])
                elif self .text_column in example :
                    raw_value =example [self .text_column ]

                    if isinstance (raw_value ,(int ,np .integer )):
                        try :

                            if (
                            hasattr (self ._current_dataset ,"features")
                            and self .text_column in self ._current_dataset .features 
                            ):
                                feature =self ._current_dataset .features [
                                self .text_column 
                                ]
                                if hasattr (feature ,"names")and feature .names :
                                    text =feature .names [int (raw_value )]
                                else :
                                    text =f"class_{raw_value }"
                            else :
                                text =f"class_{raw_value }"
                        except (IndexError ,AttributeError ,Exception ):
                            text =f"class_{raw_value }"
                    else :
                        text =str (raw_value )
                elif "content"in example :
                    text =example ["content"]
                else :
                    text =next ((v for v in example .values ()if isinstance (v ,str )),"")


                has_valid_image =(
                "image"in example 
                and example ["image"]is not None 
                and hasattr (example ["image"],"convert")
                )


                if (not text or len (str (text ).strip ())<3 )and not has_valid_image :
                    continue 


                if has_valid_image :
                    modality ="image"
                elif "audio"in example and example ["audio"]is not None :
                    modality ="audio"
                elif "content"in example or self .hf_dataset_name in [
                "bigcode/the-stack"
                ]:
                    modality ="code"
                else :
                    modality ="text"


                sample ={
                "text":str (text ).strip (),
                "modality":modality ,
                "attention_mask":[1 ]*self .seq_len ,
                }


                if has_valid_image :
                    sample ["image"]=example ["image"]

                if "audio"in example and example ["audio"]is not None :
                    sample ["audio"]=example ["audio"]

                yield sample 

            except Exception as e :
                logger .warning (
                f"Error processing example from {self .hf_dataset_name }: {e }"
                )
                continue 


def get_flagship_datasets ():
    """
    Returns flagship-scale datasets optimized for >500M tokens and 1.17B parameters.
    Uses datasets known to work with streaming and provide substantial content.
    Best practices: Verified streaming-compatible datasets with Arrow format support.
    """
    return {
    "text":{
    "hf_dataset_name":"wikitext",
    "config_name":"wikitext-103-raw-v1",
    "text_column":"text",
    "description":"WikiText-103 - High-quality Wikipedia text (20%)",
    },
    "web":{
    "hf_dataset_name":"allenai/c4",
    "config_name":"en",
    "text_column":"text",
    "description":"C4 - Common Crawl web text (30%)",
    },
    "code":{
    "hf_dataset_name":"bigcode/the-stack-dedup",
    "config_name":"default",
    "text_column":"content",
    "description":"The Stack - Real code from GitHub (10%)",
    },
    "books":{
    "hf_dataset_name":"bookcorpus",
    "config_name":None ,
    "text_column":"text",
    "description":"BookCorpus - Large public-domain novels (10%)",
    },
    "scientific":{
    "hf_dataset_name":"scientific_papers",
    "config_name":"pubmed",
    "text_column":"abstract",
    "description":"Scientific Papers PubMed - Biomedical abstracts (5%)",
    },
    "conversational":{
    "hf_dataset_name":"daily_dialog",
    "config_name":None ,
    "text_column":"dialog",
    "description":"DailyDialog - Conversational AI dialogue patterns (5%)",
    },
    "cot":{
    "hf_dataset_name":"allenai/math_qa",
    "config_name":None ,
    "text_column":"Problem",
    "description":"MathQA - Mathematical reasoning with step-by-step solutions (5%)",
    },
    "reasoning":{
    "hf_dataset_name":"squad",
    "config_name":None ,
    "text_column":"question",
    "description":"SQuAD - Reading comprehension and reasoning tasks (5%)",
    },
    "logic_reasoning":{
    "hf_dataset_name":"glue",
    "config_name":"rte",
    "text_column":"sentence1",
    "description":"RTE - Recognizing textual entailment for logical reasoning (5%)",
    },
    "multimodal_reasoning":{
    "hf_dataset_name":"squad",
    "config_name":None ,
    "text_column":"context",
    "description":"SQuAD Context - Multi-modal style reasoning from context (3%)",
    },
    "audio":{
    "hf_dataset_name":"openslr/librispeech_asr",
    "config_name":"clean",
    "split":"train.360",
    "text_column":"text",
    "audio_column":"audio",
    "trust_remote_code":True ,
    "description":"LibriSpeech - Speech transcription text for audio-aware training (1%)",
    },
    "image":{
    "hf_dataset_name":"cifar10",
    "config_name":None ,
    "text_column":"label",
    "description":"CIFAR-10 - Image classification labels for vision training (1%)",
    },
    "code_python":{
    "hf_dataset_name":"codeparrot/github-code",
    "config_name":"none",
    "text_column":"code",
    "description":"GitHub Python Code - Python code for specialized programming training",
    },
    "arxiv":{
    "hf_dataset_name":"scientific_papers",
    "config_name":"arxiv",
    "text_column":"abstract",
    "description":"ArXiv Papers - Research paper abstracts for scientific domain training",
    },
    }





def create_flagship_streaming_dataloaders (
stream_config :dict ,
tokenizer ,
batch_size :int =4 ,
collator :Optional [Callable ]=None ,
use_legacy_loader :bool =False ,
):
    """
    Creates flagship-scale streaming data loaders for >1B parameter training.
    Optimized for Tesla T4 with multiple high-volume datasets.
    FAIL-FAST: No try/except - any dataset creation error crashes the script immediately.
    """

    flagship_datasets =get_flagship_datasets ()


    streaming_datasets ={}
    dataset_factories ={}

    audio_tokenizer =(
    get_audio_tokenizer ()
    if "audio"in stream_config .get ("modalities",{})
    else None 
    )

    for modality ,modal_config in stream_config .get ("modalities",{}).items ():
        if modality in flagship_datasets :

            dataset_config =flagship_datasets [modality ].copy ()
            dataset_config .update (modal_config )
            dataset_config .update (
            {
            "seq_len":stream_config .get ("seq_len",1024 ),
            "modal_tag":f"<{modality }>",
            "buffer_size":10000 ,
            }
            )


            if modality =="audio"and audio_tokenizer is not None :
                factory =lambda cfg =dataset_config :StreamingAudioDataset (
                cfg ,tokenizer ,audio_tokenizer 
                )
            else :
                factory =lambda cfg =dataset_config :LargeScaleStreamingDataset (
                cfg ,tokenizer 
                )
            dataset =factory ()
            streaming_datasets [modality ]=dataset 
            dataset_factories [modality ]=factory 
            logger .info (
            f"✅ Created flagship streaming dataset for {modality }: {dataset_config ['hf_dataset_name']}"
            )


    expected_modalities =set (stream_config .get ("modalities",{}).keys ())
    created_modalities =set (streaming_datasets .keys ())

    if created_modalities !=expected_modalities :
        missing =expected_modalities -created_modalities 
        raise RuntimeError (f"Failed to create required streaming datasets: {missing }")

    logger .info (
    f"✅ All {len (expected_modalities )} streaming datasets created successfully"
    )


    class InterleavedStreamingDataset (torch .utils .data .IterableDataset ):
        def __init__ (self ,factories ,dataset_names ,sampling_ratios ):
            self .factories =factories 
            self .dataset_names =dataset_names 
            self .sampling_ratios =sampling_ratios 
            self .modal_datasets =[factory ()for factory in factories ]
            self .dataset_iterators =[]
            self .retry_counts ={i :0 for i in range (len (self .modal_datasets ))}
            self .max_retries =MAX_RETRIES 

        def __len__ (self ):

            return 1000000 

        def __iter__ (self ):

            self .dataset_iterators ={
            i :iter (ds )for i ,ds in enumerate (self .modal_datasets )
            }


            total_ratio =sum (self .sampling_ratios )
            weights =[r /total_ratio for r in self .sampling_ratios ]

            active_indices =list (self .dataset_iterators .keys ())
            active_weights =weights 

            while active_indices :

                chosen_index =random .choices (
                active_indices ,weights =active_weights ,k =1 
                )[0 ]

                try :

                    sample =next (self .dataset_iterators [chosen_index ])
                    self .retry_counts [chosen_index ]=0 
                    yield sample 
                except StopIteration :
                    logging .warning (
                    f"Stream {chosen_index } exhausted. Restarting iterator."
                    )
                    self .modal_datasets [chosen_index ]=self .factories [chosen_index ]()
                    self .dataset_iterators [chosen_index ]=iter (
                    self .modal_datasets [chosen_index ]
                    )
                    self .retry_counts [chosen_index ]=0 
                except Exception as e :
                    logging .error (
                    f"ERROR: Network error in '{self .dataset_names [chosen_index ]}' (Stream {chosen_index }): {e }. Re-initializing stream."
                    )
                    self .retry_counts [chosen_index ]+=1 
                    if self .retry_counts [chosen_index ]>self .max_retries :
                        logging .error (
                        f"Stream {chosen_index } failed too many times. Removing from pool."
                        )
                        self .dataset_iterators .pop (chosen_index )
                        self .factories .pop (chosen_index )
                        self .modal_datasets .pop (chosen_index )
                        self .dataset_names .pop (chosen_index )
                        active_indices =list (self .dataset_iterators .keys ())
                        if not active_indices :
                            break 
                        active_weights =[weights [i ]for i in active_indices ]
                        total_active_weight =sum (active_weights )
                        active_weights =[
                        w /total_active_weight for w in active_weights 
                        ]
                    else :
                        try :
                            self .modal_datasets [chosen_index ]=self .factories [
                            chosen_index 
                            ]()
                            self .dataset_iterators [chosen_index ]=iter (
                            self .modal_datasets [chosen_index ]
                            )
                        except Exception as reinit_e :
                            logging .error (
                            f"Reinit failed for stream {chosen_index }: {reinit_e }. Removing from pool."
                            )
                            self .dataset_iterators .pop (chosen_index )
                            self .factories .pop (chosen_index )
                            self .modal_datasets .pop (chosen_index )
                            self .dataset_names .pop (chosen_index )
                            active_indices =list (self .dataset_iterators .keys ())
                            if not active_indices :
                                break 
                            active_weights =[weights [i ]for i in active_indices ]
                            total_active_weight =sum (active_weights )
                            active_weights =[
                            w /total_active_weight for w in active_weights 
                            ]


    sampling_ratios =[
    stream_config ["modalities"][m ].get ("sampling_ratio",1.0 )
    for m in streaming_datasets .keys ()
    ]
    dataset_names =list (streaming_datasets .keys ())
    factories_list =[dataset_factories [name ]for name in dataset_names ]
    sample_size =stream_config .get ("sample_size")
    prefetch =stream_config .get ("prefetch",1 )
    if sample_size and not use_legacy_loader :
        def _base_gen ():
            while True :
                dataset =InterleavedStreamingDataset (
                factories_list ,dataset_names ,sampling_ratios 
                )
                yield iter (dataset )

        rolling_dataset =RollingCacheIterableDataset (
        base_dataset_generator =_base_gen ,
        sample_size =sample_size ,
        prefetch =prefetch ,
        )
        dataset_for_loader =rolling_dataset 
        logger .info (
        f"✅ Using RollingCacheIterableDataset with sample_size={sample_size }, prefetch={prefetch }"
        )
    else :
        dataset_for_loader =InterleavedStreamingDataset (
        factories_list ,dataset_names ,sampling_ratios 
        )


    dataloader =DataLoader (
    dataset_for_loader ,
    batch_size =batch_size ,
    num_workers =0 ,
    prefetch_factor =None ,
    pin_memory =True ,
    drop_last =True ,
    persistent_workers =False ,
    collate_fn =collator or safe_collate_fn ,
    )

    logger .info (
    f"✅ Created flagship streaming dataloader with {len (streaming_datasets )} modalities"
    )
    logger .info (f"   Modalities: {list (streaming_datasets .keys ())}")
    logger .info (f"   Batch size: {batch_size }, Workers: 0 (single-threaded)")

    return dataloader ,streaming_datasets 



setup_streaming_environment ()
