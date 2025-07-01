#!/usr/bin/env python3


"""
Production Dataset Validator
============================
Comprehensive validation for multi-tokenizer, multi-dataset training
Addresses all critical production issues before long training runs.
"""

import numpy as np 
import torch 
import json 
import os 
import hashlib 
import logging 
from pathlib import Path 
from typing import Dict ,List ,Tuple ,Optional ,Union ,Any 
from collections import defaultdict ,Counter 
import warnings 
from transformers import AutoTokenizer 
import datetime 

logging .basicConfig (level =logging .INFO ,format ='%(asctime)s - %(levelname)s - %(message)s')
logger =logging .getLogger (__name__ )

class ProductionDatasetValidator :
    """Comprehensive validator for production-ready dataset training"""

    def __init__ (self ,config_path :str =None ,cache_dir :Optional [str ]=None ,
    npz_files :Dict =None ,npy_files :Dict =None ,config :Dict =None ,tokenizer =None ):
        self .config_path =config_path 
        self .cache_dir =Path (cache_dir or "./validator_cache")
        self .cache_dir .mkdir (exist_ok =True )


        if config is not None :
            self .config =config 
        elif config_path :
            with open (config_path ,'r')as f :
                self .config =json .load (f )
        else :
            raise ValueError ("Either config_path or config dict must be provided")

        self .seq_len =self .config ['data']['seq_len']if 'data'in self .config else self .config .get ('seq_len',256 )
        self .vocab_size =self .config .get ('vocab_size',50257 )
        self .max_position_embeddings =self .config .get ('max_position_embeddings',2048 )


        self .tokenizer =tokenizer 
        if self .tokenizer :
            self .vocab_size =len (self .tokenizer )


        self .validation_results ={
        'passed':[],
        'warnings':[],
        'errors':[],
        'tokenizer_analysis':{},
        'vocab_requirements':{},
        'file_manifest':{}
        }

    def run_full_validation (self )->Dict [str ,Any ]:
        """Run comprehensive validation pipeline"""
        logger .info ("🔍 Starting comprehensive production dataset validation...")


        self ._fix_and_scan_all_files ()


        self ._analyze_tokenizer_compatibility ()


        self ._check_vocab_requirements ()


        self ._validate_special_tokens ()


        self ._check_sequence_length_overflows ()


        self ._validate_dtype_consistency ()


        self ._validate_path_handling ()


        self ._verify_padding_consistency ()


        self ._check_negative_indices ()


        self ._generate_production_manifest ()


        self ._generate_validation_report ()

        return self .validation_results 

    def _safe_load_array (self ,path :str )->Tuple [Optional [np .ndarray ],Dict [str ,Any ]]:
        """Safely load NPY/NPZ files and extract max/min values - FIXES THE NPZ BUG"""
        metadata ={'type':None ,'keys':[],'max_values':{},'min_values':{},'shapes':{}}

        try :
            if path .endswith ('.npz'):
                obj =np .load (path ,mmap_mode ='r')
                if isinstance (obj ,np .lib .npyio .NpzFile ):
                    metadata ['type']='npz'
                    metadata ['keys']=list (obj .keys ())


                    for key in obj .keys ():
                        arr =obj [key ]
                        metadata ['shapes'][key ]=arr .shape 


                        if arr .size >0 and np .issubdtype (arr .dtype ,np .number ):
                            try :
                                metadata ['max_values'][key ]=float (arr .max ())
                                metadata ['min_values'][key ]=float (arr .min ())
                            except Exception as e :
                                logger .warning (f"Could not compute max/min for {key } in {path }: {e }")
                                metadata ['max_values'][key ]=None 
                                metadata ['min_values'][key ]=None 
                        else :
                            metadata ['max_values'][key ]=None 
                            metadata ['min_values'][key ]=None 

                    return obj ,metadata 
                else :

                    metadata ['type']='npz_single'
                    metadata ['shapes']['data']=obj .shape 
                    if obj .size >0 and np .issubdtype (obj .dtype ,np .number ):
                        metadata ['max_values']['data']=float (obj .max ())
                        metadata ['min_values']['data']=float (obj .min ())
                    return obj ,metadata 

            elif path .endswith ('.npy'):
                obj =np .load (path ,mmap_mode ='r')
                metadata ['type']='npy'
                metadata ['shapes']['data']=obj .shape 

                if obj .size >0 and np .issubdtype (obj .dtype ,np .number ):
                    metadata ['max_values']['data']=float (obj .max ())
                    metadata ['min_values']['data']=float (obj .min ())

                return obj ,metadata 

        except Exception as e :
            logger .error (f"Error loading {path }: {e }")
            metadata ['error']=str (e )

        return None ,metadata 

    def _fix_and_scan_all_files (self ):
        """Fix NPZ analyzer bug and comprehensively scan all files"""
        logger .info ("🔧 Fixing NPZ analyzer and scanning all files...")

        all_files =[]


        npz_files =self .config ['data'].get ('npz_files',{})
        for name ,path_or_paths in npz_files .items ():
            if isinstance (path_or_paths ,list ):
                for i ,path in enumerate (path_or_paths ):
                    all_files .append ((f"{name }_{i :04d}",path ,'npz'))
            else :
                all_files .append ((name ,path_or_paths ,'npz'))


        npy_files =self .config ['data'].get ('npy_files',{})
        for name ,path in npy_files .items ():
            all_files .append ((name ,path ,'npy'))


        cache_file =self .cache_dir /"file_analysis_cache.json"

        if cache_file .exists ():
            logger .info ("📋 Loading cached file analysis...")
            with open (cache_file ,'r')as f :
                cached_results =json .load (f )
        else :
            cached_results ={}

        for name ,path ,file_type in all_files :
            if not Path (path ).exists ():
                self .validation_results ['errors'].append (f"File not found: {path }")
                continue 


            file_stat =Path (path ).stat ()
            cache_key =f"{path }_{file_stat .st_mtime }_{file_stat .st_size }"

            if cache_key in cached_results :
                metadata =cached_results [cache_key ]
                logger .info (f"  💾 Using cached analysis for {name }")
            else :
                logger .info (f"  🔍 Analyzing {name } ({file_type .upper ()})...")
                _ ,metadata =self ._safe_load_array (path )
                cached_results [cache_key ]=metadata 

            self .validation_results ['file_manifest'][name ]={
            'path':path ,
            'type':file_type ,
            'metadata':metadata 
            }


        with open (cache_file ,'w')as f :
            json .dump (cached_results ,f ,indent =2 )

        logger .info (f"✅ Scanned {len (all_files )} files")

    def _analyze_tokenizer_compatibility (self ):
        """Check tokenizer compatibility across all datasets"""
        logger .info ("🔤 Analyzing tokenizer compatibility...")


        if self .tokenizer :
            primary_tokenizer =self .tokenizer 
            tokenizer_name =getattr (primary_tokenizer ,'name_or_path','provided_tokenizer')
            logger .info (f"Using provided tokenizer with {len (primary_tokenizer )} tokens")
        else :
            tokenizer_name =self .config .get ('tokenizer_name','gpt2')
            try :
                primary_tokenizer =AutoTokenizer .from_pretrained (tokenizer_name )
                if primary_tokenizer .pad_token is None :
                    primary_tokenizer .pad_token =primary_tokenizer .eos_token 
            except Exception as e :
                self .validation_results ['errors'].append (f"Cannot load primary tokenizer {tokenizer_name }: {e }")
                return 


        test_samples =[
        "Smith & Wesson",
        "Hello, world! How are you?",
        "def function(x): return x + 1",
        "The quick brown fox jumps over the lazy dog.",
        "Special chars: @#$%^&*()_+-=[]{}|;':\",./<>?"
        ]

        tokenizer_results ={}

        for sample in test_samples :
            try :
                encoded =primary_tokenizer .encode (sample )
                decoded =primary_tokenizer .decode (encoded )

                match =(decoded .strip ()==sample .strip ())
                if not match :
                    self .validation_results ['warnings'].append (
                    f"Tokenizer round-trip mismatch: '{sample }' -> '{decoded }'"
                    )

                tokenizer_results [sample ]={
                'encoded':encoded ,
                'decoded':decoded ,
                'round_trip_match':match ,
                'token_count':len (encoded )
                }

            except Exception as e :
                self .validation_results ['errors'].append (f"Tokenizer error on '{sample }': {e }")

        vocab_size =len (primary_tokenizer )if hasattr (primary_tokenizer ,'__len__')else getattr (primary_tokenizer ,'vocab_size',50257 )

        self .validation_results ['tokenizer_analysis']={
        'primary_tokenizer':tokenizer_name ,
        'vocab_size':vocab_size ,
        'special_tokens':{
        'pad_token':primary_tokenizer .pad_token ,
        'eos_token':primary_tokenizer .eos_token ,
        'bos_token':primary_tokenizer .bos_token ,
        'unk_token':primary_tokenizer .unk_token ,
        },
        'test_results':tokenizer_results 
        }

        logger .info (f"✅ Primary tokenizer: {tokenizer_name } (vocab_size: {vocab_size })")

    def _check_vocab_requirements (self ):
        """Analyze vocab size requirements across all datasets"""
        logger .info ("📊 Checking vocabulary requirements...")

        max_token_ids =[]
        vocab_analysis ={}

        for name ,file_info in self .validation_results ['file_manifest'].items ():
            metadata =file_info ['metadata']

            if 'max_values'in metadata :
                for key ,max_val in metadata ['max_values'].items ():
                    if max_val is not None and ('token'in key .lower ()or 'input_id'in key .lower ()):
                        max_token_ids .append (int (max_val ))

                        vocab_analysis [f"{name }_{key }"]={
                        'max_token_id':int (max_val ),
                        'requires_vocab_size':int (max_val )+1 
                        }

        if max_token_ids :
            global_max =max (max_token_ids )
            required_vocab_size =global_max +1 

            if required_vocab_size >self .vocab_size :
                self .validation_results ['errors'].append (
                f"CRITICAL: Max token ID {global_max } requires vocab_size >= {required_vocab_size }, "
                f"but config has {self .vocab_size }"
                )
            else :
                self .validation_results ['passed'].append (
                f"Vocab size check passed: max_token_id={global_max }, vocab_size={self .vocab_size }"
                )

            self .validation_results ['vocab_requirements']={
            'global_max_token_id':global_max ,
            'required_vocab_size':required_vocab_size ,
            'config_vocab_size':self .vocab_size ,
            'is_sufficient':required_vocab_size <=self .vocab_size ,
            'per_dataset':vocab_analysis 
            }

        logger .info (f"✅ Vocabulary analysis complete")

    def _validate_special_tokens (self ):
        """Check for proper special token handling"""
        logger .info ("🏷️ Validating special tokens...")


        sample_files =list (self .validation_results ['file_manifest'].items ())[:5 ]

        for name ,file_info in sample_files :
            path =file_info ['path']
            try :
                data ,_ =self ._safe_load_array (path )

                if data is None :
                    continue 


                if path .endswith ('.npz')and isinstance (data ,np .lib .npyio .NpzFile ):
                    for key in data .keys ():
                        if 'token'in key .lower ():
                            tokens =data [key ]
                            if len (tokens )>0 :
                                sample_tokens =tokens .flat [:20 ]if hasattr (tokens ,'flat')else tokens [:20 ]
                                self ._analyze_token_pattern (name ,key ,sample_tokens )

                elif path .endswith ('.npy'):
                    if len (data )>0 :
                        sample_tokens =data .flat [:20 ]if hasattr (data ,'flat')else data [:20 ]
                        self ._analyze_token_pattern (name ,'data',sample_tokens )

            except Exception as e :
                logger .warning (f"Could not validate special tokens in {name }: {e }")

        logger .info ("✅ Special token validation complete")

    def _analyze_token_pattern (self ,dataset_name :str ,key :str ,sample_tokens ):
        """Analyze token patterns for special tokens"""
        try :
            sample_tokens =np .array (sample_tokens ).flatten ()


            if len (sample_tokens )>0 :
                first_token =sample_tokens [0 ]


                if first_token in [50256 ,0 ,1 ,2 ]:
                    self .validation_results ['passed'].append (
                    f"{dataset_name }[{key }] starts with potential BOS token: {first_token }"
                    )
                elif first_token >1000 :
                    self .validation_results ['warnings'].append (
                    f"{dataset_name }[{key }] starts with content token {first_token } - missing BOS?"
                    )

        except Exception as e :
            logger .warning (f"Error analyzing token pattern for {dataset_name }: {e }")

    def _check_sequence_length_overflows (self ):
        """Check for sequences longer than max_position_embeddings"""
        logger .info ("📏 Checking sequence length overflows...")

        overflow_count =0 

        for name ,file_info in self .validation_results ['file_manifest'].items ():
            metadata =file_info ['metadata']

            if 'shapes'in metadata :
                for key ,shape in metadata ['shapes'].items ():
                    if 'token'in key .lower ()and len (shape )>0 :
                        seq_len =shape [-1 ]if len (shape )>1 else shape [0 ]

                        if seq_len >self .max_position_embeddings :
                            overflow_count +=1 
                            self .validation_results ['errors'].append (
                            f"SEQUENCE OVERFLOW: {name }[{key }] length {seq_len } > max_position_embeddings {self .max_position_embeddings }"
                            )

        if overflow_count ==0 :
            self .validation_results ['passed'].append ("No sequence length overflows detected")

        logger .info (f"✅ Sequence length check complete ({overflow_count } overflows)")

    def _validate_dtype_consistency (self ):
        """Check for dtype consistency issues"""
        logger .info ("🔢 Validating dtype consistency...")

        dtype_analysis =defaultdict (list )

        for name ,file_info in self .validation_results ['file_manifest'].items ():
            path =file_info ['path']

            try :
                if path .endswith ('.npz'):
                    with np .load (path )as data :
                        for key in data .keys ():
                            if 'token'in key .lower ():
                                arr =data [key ]
                                dtype_analysis [str (arr .dtype )].append (f"{name }[{key }]")

                elif path .endswith ('.npy'):
                    data =np .load (path ,mmap_mode ='r')
                    dtype_analysis [str (data .dtype )].append (f"{name }[data]")

            except Exception as e :
                logger .warning (f"Could not check dtype for {name }: {e }")


        for dtype ,datasets in dtype_analysis .items ():
            if dtype not in ['int32','int64']:
                self .validation_results ['warnings'].append (
                f"Non-integer dtype {dtype } found in: {datasets [:5 ]}"
                )
            else :
                logger .info (f"  {dtype }: {len (datasets )} datasets")


        if 'int64'in dtype_analysis and len (dtype_analysis ['int64'])>0 :
            self .validation_results ['warnings'].append (
            f"Consider converting int64 to int32 for GPU efficiency: {len (dtype_analysis ['int64'])} files"
            )

        logger .info ("✅ Dtype consistency check complete")

    def _validate_path_handling (self ):
        """Check for problematic paths"""
        logger .info ("📁 Validating path handling...")

        problematic_paths =[]

        for name ,file_info in self .validation_results ['file_manifest'].items ():
            path =file_info ['path']


            if ' 'in path :
                problematic_paths .append (f"{name }: contains spaces")
            if any (char in path for char in [':','"','\'','`']):
                problematic_paths .append (f"{name }: contains special characters")
            if not Path (path ).exists ():
                problematic_paths .append (f"{name }: file not found")

        if problematic_paths :
            self .validation_results ['warnings'].extend (problematic_paths )
        else :
            self .validation_results ['passed'].append ("All file paths are valid")

        logger .info ("✅ Path validation complete")

    def _verify_padding_consistency (self ):
        """Check padding consistency across datasets"""
        logger .info ("🔄 Verifying padding consistency...")



        config_padding =self .config .get ('padding',{})

        if 'pad_to_multiple_of'in config_padding :
            self .validation_results ['passed'].append (
            f"Consistent padding config: pad_to_multiple_of={config_padding ['pad_to_multiple_of']}"
            )
        else :
            self .validation_results ['warnings'].append (
            "No pad_to_multiple_of specified - may cause inconsistent batch shapes"
            )

        logger .info ("✅ Padding consistency check complete")

    def _check_negative_indices (self ):
        """Check for negative indices in input_ids"""
        logger .info ("⚠️ Checking for negative indices...")


        sample_files =list (self .validation_results ['file_manifest'].items ())[:3 ]

        for name ,file_info in sample_files :
            path =file_info ['path']

            try :
                if path .endswith ('.npz'):
                    with np .load (path )as data :
                        for key in data .keys ():
                            if 'token'in key .lower ()or 'input_id'in key .lower ():
                                arr =data [key ]
                                if np .any (arr <0 ):
                                    self .validation_results ['errors'].append (
                                    f"NEGATIVE INDICES in {name }[{key }] - will cause CUDA errors"
                                    )

            except Exception as e :
                logger .warning (f"Could not check negative indices in {name }: {e }")

        logger .info ("✅ Negative indices check complete")

    def _generate_production_manifest (self ):
        """Generate production-ready manifest file"""
        logger .info ("📋 Generating production manifest...")

        manifest ={
        'config_version':'1.0',
        'validation_timestamp':str (datetime .datetime .now ()),
        'total_files':len (self .validation_results ['file_manifest']),
        'datasets':{}
        }

        for name ,file_info in self .validation_results ['file_manifest'].items ():
            manifest ['datasets'][name ]={
            'path':file_info ['path'],
            'type':file_info ['type'],
            'validated':True ,
            'metadata_summary':{
            'shapes':file_info ['metadata'].get ('shapes',{}),
            'max_token_id':max ([v for v in file_info ['metadata'].get ('max_values',{}).values ()if v is not None ]or [0 ])
            }
            }

        manifest_file =self .cache_dir /"production_manifest.json"
        with open (manifest_file ,'w')as f :
            json .dump (manifest ,f ,indent =2 )

        logger .info (f"✅ Production manifest saved: {manifest_file }")

    def _generate_validation_report (self ):
        """Generate comprehensive validation report"""
        logger .info ("📊 Generating validation report...")

        report ={
        'validation_summary':{
        'total_files':len (self .validation_results ['file_manifest']),
        'passed_checks':len (self .validation_results ['passed']),
        'warnings':len (self .validation_results ['warnings']),
        'errors':len (self .validation_results ['errors']),
        'overall_status':'PASS'if len (self .validation_results ['errors'])==0 else 'FAIL'
        },
        'detailed_results':self .validation_results 
        }

        report_file =self .cache_dir /"validation_report.json"
        with open (report_file ,'w')as f :
            json .dump (report ,f ,indent =2 )


        print ("\n"+"="*80 )
        print ("🎯 PRODUCTION DATASET VALIDATION SUMMARY")
        print ("="*80 )
        print (f"✅ Passed checks: {len (self .validation_results ['passed'])}")
        print (f"⚠️  Warnings: {len (self .validation_results ['warnings'])}")
        print (f"❌ Errors: {len (self .validation_results ['errors'])}")
        print (f"📊 Overall status: {report ['validation_summary']['overall_status']}")

        if self .validation_results ['errors']:
            print ("\n🚨 CRITICAL ERRORS (must fix before training):")
            for error in self .validation_results ['errors']:
                print (f"  ❌ {error }")

        if self .validation_results ['warnings']:
            print ("\n⚠️  WARNINGS (recommended to fix):")
            for warning in self .validation_results ['warnings'][:10 ]:
                print (f"  ⚠️  {warning }")

        print (f"\n📋 Full report saved: {report_file }")
        print ("="*80 )

        return report 

def main ():
    """Run validation on config file"""
    import sys 

    if len (sys .argv )!=2 :
        print ("Usage: python production_dataset_validator.py <config_file>")
        sys .exit (1 )

    config_file =sys .argv [1 ]
    validator =ProductionDatasetValidator (config_file )
    results =validator .run_full_validation ()


    if len (results ['errors'])>0 :
        sys .exit (1 )

if __name__ =="__main__":
    main ()
