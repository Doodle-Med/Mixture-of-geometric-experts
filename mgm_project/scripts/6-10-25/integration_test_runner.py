#!/usr/bin/env python3


"""Integration test runner for the MGM training pipeline."""


import os 
import sys 
import json 
import time 
import logging 
import argparse 
from pathlib import Path 
from copy import deepcopy 
import math 


project_root =Path (__file__ ).resolve ().parent 
sys .path .insert (0 ,str (project_root ))



try :
    import train_geometric_model_v2 as trainer 
    from run_flagship_production import FlagshipTrainingOrchestrator 
except ImportError as e :
    print (
    f"FATAL: Could not import core MGM modules. Ensure this script is in the correct directory."
    )
    print (f"Error: {e }")
    sys .exit (1 )



logging .basicConfig (
level =logging .INFO ,
format ="%(asctime)s - %(levelname)s - [TestRunner] - %(message)s",
stream =sys .stdout ,
)
logger =logging .getLogger (__name__ )


def create_argument_parser ():
    """Creates a comprehensive argument parser for all test configurations."""
    parser =argparse .ArgumentParser (
    description ="MGM Comprehensive Integration Test Runner",
    formatter_class =argparse .RawDescriptionHelpFormatter ,
    )


    parser .add_argument (
    "--base-config",
    type =str ,
    default ="production_flagship_config.json",
    help ="Path to the base JSON configuration file to modify.",
    )
    parser .add_argument (
    "--skip-setup",
    action ="store_true",
    help ="Skip running the setup_env.sh script.",
    )


    parser .add_argument (
    "--stage-steps",
    type =int ,
    default =10 ,
    help ="Run each training stage for this many steps.",
    )
    parser .add_argument (
    "--hidden-dim",type =int ,default =128 ,help ="Set the model hidden dimension."
    )
    parser .add_argument (
    "--seq-len",type =int ,default =32 ,help ="Set the sequence length for training."
    )


    modality_group =parser .add_mutually_exclusive_group ()
    modality_group .add_argument (
    "--only-audio",action ="store_true",help ="Test with only audio datasets."
    )
    modality_group .add_argument (
    "--only-vision",action ="store_true",help ="Test with only vision datasets."
    )
    modality_group .add_argument (
    "--only-text",action ="store_true",help ="Test with only text-based datasets."
    )


    parser .add_argument ("--amp-off",action ="store_true",help ="Disable AMP and FP16.")
    parser .add_argument (
    "--amp-on",
    action ="store_true",
    help ="Force enable AMP and FP16 (requires GPU).",
    )
    parser .add_argument (
    "--flash-attention-off",action ="store_true",help ="Disable Flash Attention."
    )
    parser .add_argument (
    "--flash-attention-on",
    action ="store_true",
    help ="Force enable Flash Attention (requires GPU).",
    )
    parser .add_argument (
    "--gradient-checkpointing-on",
    action ="store_true",
    help ="Enable activation checkpointing",
    )
    parser .add_argument (
    "--gradient-checkpointing-off",
    action ="store_true",
    help ="Disable activation checkpointing",
    )
    parser .add_argument (
    "--ppo-off",action ="store_true",help ="Disable the PPO/Verifier stage."
    )

    parser .add_argument ("--convert-from-dense",action ="store_true",
    help ="Clone weights from a dense LM into every expert (+𝒩(0,0.02 I)).")
    parser .add_argument ("--dense-init",type =str ,nargs ="?",const ="gpt2-xl",default ="",
    help ="🤗 hub ID or local path to the dense model (default gpt2-xl).")
    parser .add_argument ("--ppo-kl-coef",type =float ,default =0.02 ,
    help ="β for KL(π‖π₀) regulariser; auto-tuned at runtime.")
    parser .add_argument ("--freeze-emb-steps",type =int ,default =3000 ,
    help ="Number of global steps to keep token/pos embeddings frozen.")
    parser .add_argument ("--ppo-target-kl",type =float ,default =0.05 ,
    help ="Target KL divergence for PPO adaptive β controller")



    parser .add_argument (
    "--experts-num",
    type =int ,
    default =None ,
    help ="Set the number of experts. Supports any positive integer.",
    )

    parser .add_argument (
    "--k-experts",
    type =int ,
    default =None ,
    help ="Set the number of experts to activate (top-k routing).",
    )


    parser .add_argument (
    "--num-layers",
    type =int ,
    default =None ,
    help ="Set the number of transformer layers.",
    )

    parser .add_argument (
    "--num-heads",
    type =int ,
    default =None ,
    help ="Set the number of attention heads.",
    )


    parser .add_argument (
    "--learning-rate",
    type =float ,
    default =None ,
    help ="Set the learning rate.",
    )

    parser .add_argument (
    "--grad-accumulation",
    type =int ,
    default =None ,
    help ="Set gradient accumulation steps.",
    )

    parser .add_argument (
    "--warmup-steps",
    type =int ,
    default =None ,
    help ="Set learning rate warmup steps.",
    )

    parser .add_argument (
    "--save-steps",
    type =int ,
    default =None ,
    help ="Set checkpoint save interval.",
    )

    parser .add_argument (
    "--eval-steps",
    type =int ,
    default =None ,
    help ="Set evaluation interval.",
    )

    parser .add_argument (
    "--logging-steps",
    type =int ,
    default =None ,
    help ="Set logging interval.",
    )

    parser .add_argument (
    "--max-grad-norm",
    type =float ,
    default =None ,
    help ="Set gradient clipping norm.",
    )


    parser .add_argument (
    "--config",
    type =str ,
    default =None ,
    help ="Path to configuration file (alternative to --base-config).",
    )

    parser .add_argument (
    "--batch-size",type =int ,default =None ,help ="Override training batch size"
    )

    parser .add_argument (
    "--sample-size",
    type =int ,
    default =None ,
    help ="Optional limit on number of samples to load from each dataset.",
    )

    parser .add_argument (
    "--validation-steps",
    type =int ,
    default =None ,
    help ="Number of batches to run during each validation phase",
    )

    parser .add_argument (
    "--nuanced-routing",
    action ="store_true",
    help ="Enable revolutionary nuanced geometric routing for sophisticated understanding",
    )
    parser.add_argument(
    "--analogy-off",
    action="store_true",
    help="Disable AnalogyReasoner module",
    )

    parser .add_argument (
    "--prefetch",
    type =int ,
    default =1 ,
    help ="number of caches to pre-load ahead of time",
    )


    parser .add_argument (
    "--memory-slots",
    type =int ,
    default =None ,
    help ="Set number of memory slots (lower = less memory usage)",
    )

    parser .add_argument (
    "--recursion-steps",
    type =int ,
    default =None ,
    help ="Set number of recursion steps (lower = less memory usage)",
    )

    parser .add_argument (
    "--memory-width",
    type =int ,
    default =None ,
    help ="Set memory width (lower = less memory usage)",
    )


    parser .add_argument (
    "--dataset-conversational",
    action ="store_true",
    help ="Include conversational datasets",
    )

    parser .add_argument (
    "--dataset-reasoning",
    action ="store_true",
    help ="Include reasoning datasets",
    )

    parser .add_argument (
    "--dataset-code",
    action ="store_true",
    help ="Include code datasets",
    )

    parser .add_argument (
    "--dataset-scientific",
    action ="store_true",
    help ="Include scientific datasets",
    )

    parser .add_argument (
    "--dataset-web",
    action ="store_true",
    help ="Include web datasets",
    )

    parser .add_argument (
    "--dataset-books",
    action ="store_true",
    help ="Include books datasets",
    )

    parser .add_argument (
    "--dataset-cot",
    action ="store_true",
    help ="Include chain-of-thought datasets",
    )

    parser .add_argument (
    "--dataset-logic_reasoning",
    action ="store_true",
    help ="Include logic reasoning datasets",
    )

    parser .add_argument (
    "--dataset-math",
    action ="store_true",
    help ="Include math dataset in training",
    )

    parser .add_argument (
    "--dataset-wikitext",
    action ="store_true",
    help ="Include wikitext dataset in training",
    )


    parser .add_argument (
    "--curvature-batches",
    type =int ,
    default =None ,
    help ="Number of batches for curvature calibration",
    )

    parser .add_argument (
    "--debug-checks",
    action ="store_true",
    help ="enable NaN/Inf debug assertions during training",
    )

    parser .add_argument (
    "--legacy-loader",
    action ="store_true",
    help ="use legacy streaming loader instead of rolling cache",
    )

    return parser 


def patch_config (base_config ,args ):
    """
    Modifies the base configuration in-memory based on CLI arguments.
    This is the "monkey-patching" engine.
    """
    logger .info ("🔧 Patching configuration based on command-line arguments...")
    config =deepcopy (base_config )


    if args .convert_from_dense :
        config ['convert_from_dense']=True 
        logger .info ("Enabled dense-to-MoE conversion.")

    config .setdefault ('model',{})
    config .setdefault ('training',{})

    if args .dense_init :
        config ['model']['dense_init']=args .dense_init 
        logger .info (f"Dense warm-start from {args .dense_init }")

    config ['training']['ppo_kl_coef']=args .ppo_kl_coef 
    config ['training']['freeze_emb_steps']=args .freeze_emb_steps 
    config ['training']['ppo_target_kl']=args .ppo_target_kl 
    logger .info (
    f"KL β → {args .ppo_kl_coef }   target_KL → {args .ppo_target_kl }   freeze_emb_steps → {args .freeze_emb_steps }"
    )

    if 'router_temperature_decay_steps'not in config ['model']:
        config ['model']['router_temperature_decay_steps']=10000 


    config ["training"]["max_steps"]=args .stage_steps 
    config ["training"]["warmup_steps"]=min (2 ,args .stage_steps )
    config ["training"]["logging_steps"]=1 
    config ["training"]["save_steps"]=args .stage_steps +10 


    if args .validation_steps is not None :
        config ["training"]["validation_max_steps"]=args .validation_steps 
        logger .info (f"Set validation_max_steps to {args .validation_steps }")
    else :
        sample_size =args .sample_size if args .sample_size else 1000 
        auto_steps =max (10 ,min (sample_size ,50 ))
        config ["training"]["validation_max_steps"]=auto_steps 
        logger .info (
        f"Auto-calculated validation_max_steps: {auto_steps } (sample_size={sample_size })"
        )


    if args .nuanced_routing :
        config ["model"]["use_nuanced_routing"]=True
        config ["model"]["num_concept_groups"]=4
        logger .info ("🧠 ENABLED: Revolutionary Nuanced Geometric Routing for sophisticated understanding!")

    if args.analogy_off:
        config["model"]["use_analogy_reasoner"] = False
        logger.info("Disabled AnalogyReasoner module")


    if args .eval_steps is None :
        val_max_steps =config ["training"]["validation_max_steps"]
        smart_eval_steps =min (args .stage_steps //4 ,val_max_steps *2 ,25 )
        config ["training"]["eval_steps"]=max (smart_eval_steps ,5 )
        logger .info (
        f"Auto-calculated optimal eval_steps: {config ['training']['eval_steps']} (validates every {config ['training']['eval_steps']} training steps)"
        )


    for stage_name ,stage_config in config .get ("training_stages",{}).items ():
        stage_config ["epochs"]=1 

        if "optimizer"not in stage_config :
            stage_config ["optimizer"]={}
    logger .info (f"Set max_steps to {args .stage_steps } and all stage epochs to 1.")

    if args .batch_size :
        config ["training"]["batch_size"]=args .batch_size 
        logger .info (f"Overrode batch_size to {args .batch_size }")

    config ["model"]["hidden_dim"]=args .hidden_dim 
    config ["model"]["input_dim"]=args .hidden_dim 
    config ["model"]["output_dim"]=args .hidden_dim 
    config ["model"]["memory_width"]=args .hidden_dim 
    logger .info (
    f"Set hidden_dim to {args .hidden_dim }; input, output and memory widths match hidden_dim"
    )

    config ["data"]["seq_len"]=args .seq_len 
    config ["streaming"]["seq_len"]=args .seq_len 
    logger .info (f"Set sequence_length to {args .seq_len }.")


    all_modalities =deepcopy (config ["streaming"]["modalities"])
    if args .only_audio :
        config ["streaming"]["modalities"]={
        k :v for k ,v in all_modalities .items ()if "audio"in k 
        }
        logger .info ("Enabled ONLY audio data modalities.")
    elif args .only_vision :
        config ["streaming"]["modalities"]={
        k :v for k ,v in all_modalities .items ()if "image"in k 
        }
        logger .info ("Enabled ONLY vision data modalities.")
    elif args .only_text :
        config ["streaming"]["modalities"]={
        k :v 
        for k ,v in all_modalities .items ()
        if "audio"not in k and "image"not in k 
        }
        logger .info ("Enabled ONLY text-based data modalities.")


    if args .amp_off :
        config ["training"]["use_amp"]=False 
        config ["training"]["fp16"]=False 
        logger .info ("Disabled AMP and FP16.")
    if args .amp_on :
        config ["training"]["use_amp"]=True 
        config ["training"]["fp16"]=True 
        logger .info ("Force-enabled AMP and FP16 (requires GPU).")

    if args .flash_attention_off :
        config ["training"]["use_flash_attention"]=False 
        logger .info ("Disabled Flash Attention.")
    if args .flash_attention_on :
        config ["training"]["use_flash_attention"]=True 
        logger .info ("Force-enabled Flash Attention (requires GPU).")

    if args .ppo_off :
        for stage in config ["training_stages"].values ():
            if "optimizer"in stage :
                stage ["optimizer"]["reward_lr"]=0.0 
        logger .info ("Disabled PPO/Verifier by setting reward_lr to 0.")


    if args .experts_num :
        all_manifold_types =[
        "euclidean",
        "hyperbolic",
        "spherical",
        "poincare",
        "simplex",
        "complex",
        "lorentzian",
        "product",
        ]
        repeat =(args .experts_num +len (all_manifold_types )-1 )//len (
        all_manifold_types 
        )
        config ["model"]["manifolds"]=(all_manifold_types *repeat )[:args .experts_num ]
        config ["model"]["num_experts"]=args .experts_num 
        if config ["model"].get ("k",4 )>args .experts_num :
            config ["model"]["k"]=args .experts_num 
            logger .info (f"Adjusted gating k to {args .experts_num } due to fewer experts")
        logger .info (f"Set number of experts to {config ['model']['num_experts']}.")

    E =config ['model'].get ('num_experts',len (config ['model'].get ('manifolds',[])))
    if E >0 :
        cyclic =["euclidean","hyperbolic","spherical"]*math .ceil (E /3 )
        config ['model']['manifolds']=cyclic [:E ]
        logger .info (f"🔄 Cycled manifold list to ensure mixed-curvature prior across {E } experts.")

    if args .sample_size :
        config ["streaming"]["sample_size"]=args .sample_size 
        logger .info (f"Limited dataset sample size to {args .sample_size }")


        config ["use_streaming"]=True 

    config ["streaming"]["prefetch"]=args .prefetch 
    logger .info (f"Set prefetch to {args .prefetch }")

    if args .debug_checks :
        config ["training"]["debug_checks"]=True 
        logger .info ("Enabled debug NaN checks")

    if args .legacy_loader :
        config ["streaming"]["use_legacy_loader"]=True 
        logger .info ("Using legacy streaming loader")


    if args .memory_slots is not None :
        config ["model"]["memory_slots"]=args .memory_slots 
        logger .info (f"Set memory slots to {args .memory_slots }")

    if args .recursion_steps is not None :
        config ["model"]["recursion_steps"]=args .recursion_steps 
        logger .info (f"Set recursion steps to {args .recursion_steps }")

    if args .memory_width is not None :
        config ["model"]["memory_width"]=args .memory_width 
        logger .info (f"Set memory width to {args .memory_width }")


    if args .k_experts is not None :
        config ["model"]["k"]=args .k_experts 
        logger .info (f"Set k-experts (top-k routing) to {args .k_experts }")

    if args .num_layers is not None :
        config ["model"]["num_layers"]=args .num_layers 
        logger .info (f"Set number of layers to {args .num_layers }")

    if args .num_heads is not None :
        config ["model"]["num_heads"]=args .num_heads 
        logger .info (f"Set number of attention heads to {args .num_heads }")


    if args .learning_rate is not None :
        config ["training"]["learning_rate"]=args .learning_rate 

        for stage_name ,stage_config in config .get ("training_stages",{}).items ():
            if "optimizer"not in stage_config :
                stage_config ["optimizer"]={}
            stage_config ["optimizer"]["lr"]=args .learning_rate 
            stage_config ["learning_rate"]=args .learning_rate 
        logger .info (f"Set learning rate to {args .learning_rate }")

    if args .grad_accumulation is not None :
        config ["training"]["gradient_accumulation_steps"]=args .grad_accumulation 
        logger .info (f"Set gradient accumulation steps to {args .grad_accumulation }")

    if args .warmup_steps is not None :
        config ["training"]["warmup_steps"]=args .warmup_steps 
        logger .info (f"Set warmup steps to {args .warmup_steps }")

    if args .save_steps is not None :
        config ["training"]["save_steps"]=args .save_steps 
        logger .info (f"Set save steps to {args .save_steps }")

    if args .eval_steps is not None :
        config ["training"]["eval_steps"]=args .eval_steps 
        logger .info (f"Set eval steps to {args .eval_steps }")

    if args .logging_steps is not None :
        config ["training"]["logging_steps"]=args .logging_steps 
        logger .info (f"Set logging steps to {args .logging_steps }")

    if args .max_grad_norm is not None :
        config ["training"]["max_grad_norm"]=args .max_grad_norm 
        logger .info (f"Set max gradient norm to {args .max_grad_norm }")


    if args .config is not None :
        logger .info (f"Note: --config {args .config } specified (alternative to --base-config)")


    dataset_flags =[
    args .dataset_conversational ,args .dataset_reasoning ,args .dataset_code ,
    args .dataset_scientific ,args .dataset_web ,args .dataset_books ,
    args .dataset_cot ,args .dataset_logic_reasoning ,args .dataset_math ,args .dataset_wikitext 
    ]

    if any (dataset_flags ):

        config ["streaming"]["dataset_selection"]={
        "conversational":args .dataset_conversational ,
        "reasoning":args .dataset_reasoning ,
        "code":args .dataset_code ,
        "scientific":args .dataset_scientific ,
        "web":args .dataset_web ,
        "books":args .dataset_books ,
        "cot":args .dataset_cot ,
        "logic_reasoning":args .dataset_logic_reasoning ,
        "math":args .dataset_math ,
        "wikitext":args .dataset_wikitext 
        }


        selected_modalities ={}
        all_modalities =config ["streaming"]["modalities"]

        if args .dataset_conversational and "conversational"in all_modalities :
            selected_modalities ["conversational"]=all_modalities ["conversational"]
        if args .dataset_conversational and "daily_dialog"in all_modalities :
            selected_modalities ["daily_dialog"]=all_modalities ["daily_dialog"]

        if args .dataset_reasoning and "reasoning"in all_modalities :
            selected_modalities ["reasoning"]=all_modalities ["reasoning"]
        if args .dataset_reasoning and "squad"in all_modalities :
            selected_modalities ["squad"]=all_modalities ["squad"]
        if args .dataset_reasoning and "logic_reasoning"in all_modalities :
            selected_modalities ["logic_reasoning"]=all_modalities ["logic_reasoning"]
        if args .dataset_reasoning and "cot"in all_modalities :
            selected_modalities ["cot"]=all_modalities ["cot"]

        if args .dataset_code and "code"in all_modalities :
            selected_modalities ["code"]=all_modalities ["code"]
        if args .dataset_code and "code_python"in all_modalities :
            selected_modalities ["code_python"]=all_modalities ["code_python"]

        if args .dataset_scientific and "scientific"in all_modalities :
            selected_modalities ["scientific"]=all_modalities ["scientific"]
        if args .dataset_scientific and "arxiv"in all_modalities :
            selected_modalities ["arxiv"]=all_modalities ["arxiv"]

        if args .dataset_web and "web"in all_modalities :
            selected_modalities ["web"]=all_modalities ["web"]

        if args .dataset_books and "books"in all_modalities :
            selected_modalities ["books"]=all_modalities ["books"]

        if args .dataset_cot and "cot"in all_modalities :
            selected_modalities ["cot"]=all_modalities ["cot"]

        if args .dataset_logic_reasoning and "logic_reasoning"in all_modalities :
            selected_modalities ["logic_reasoning"]=all_modalities ["logic_reasoning"]

        if args .dataset_math and "math"in all_modalities :
            selected_modalities ["math"]=all_modalities ["math"]

        if args .dataset_wikitext and "wikitext"in all_modalities :
            selected_modalities ["wikitext"]=all_modalities ["wikitext"]

        config ["streaming"]["modalities"]=selected_modalities 
        logger .info (f"Selected datasets: {list (selected_modalities .keys ())}")


    if args .curvature_batches is not None :
        if "training_stages"in config and "curvature_calibration"in config ["training_stages"]:
            config ["training_stages"]["curvature_calibration"]["smart_calibration_batches"]=args .curvature_batches 
            config ["training_stages"]["curvature_calibration"]["max_steps"]=args .curvature_batches 
            logger .info (f"Set curvature calibration to {args .curvature_batches } batches")



    config ["training"]["save_steps"]=args .stage_steps 
    logger .info (f"Set save_steps to {args .stage_steps } to save only final models")



    config ["model"]["enable_vision"]=any (
    "image"in k for k in config ["streaming"]["modalities"]
    )
    config ["model"]["enable_audio"]=any (
    "audio"in k for k in config ["streaming"]["modalities"]
    )




    config ['training'].setdefault ('freeze_curvature_steps',10_000 )
    config ['training'].setdefault ('freeze_router_steps',10_000 )

    config .setdefault ('router',{})
    config ['router'].setdefault ('softmax_temp',{
    'schedule':{
    'type':'cosine',
    'start':2.0 ,
    'end':1.0 ,
    'steps':5_000 ,
    }
    })
    logger .info ('Applied freeze_curvature_steps, freeze_router_steps and router temperature schedule overrides.')

    logger .info ("✅ Configuration patching complete.")
    return config 


def main ():
    """Main execution function."""
    parser =create_argument_parser ()
    args =parser .parse_args ()

    use_ckpt =args .gradient_checkpointing_on and not args .gradient_checkpointing_off 


    if not args .skip_setup :
        logger .info ("🏃 Running environment setup script...")
        setup_script_path =project_root /"setup_env.sh"
        if not setup_script_path .exists ():
            logger .error (f"Setup script not found at {setup_script_path }")
            sys .exit (1 )
        os .system (f"bash {setup_script_path }")
    else :
        logger .info ("Skipping environment setup as requested.")



    config_file =args .config if args .config else args .base_config 
    base_config_path =project_root /config_file 
    if not base_config_path .exists ():
        logger .error (f"Configuration file not found: {base_config_path }")
        sys .exit (1 )

    with open (base_config_path ,"r")as f :
        base_config =json .load (f )

    test_config =patch_config (base_config ,args )


    test_config ["training"]["gradient_checkpointing"]=use_ckpt 


    patched_config_path =project_root /"test_config_patched.json"
    with open (patched_config_path ,"w")as f :
        json .dump (test_config ,f ,indent =2 )
    logger .info (f"Patched test configuration saved to: {patched_config_path }")






    logger .info ("🚀 LAUNCHING MGM TRAINING PIPELINE WITH TEST CONFIGURATION 🚀")

    try :


        orchestrator =FlagshipTrainingOrchestrator (patched_config_path )
        orchestrator .config =test_config 


        orchestrator .env_setup .setup_environment ()
        orchestrator .setup_monitoring ()


        orchestrator .run_training ()

        logger .info ("🎉 INTEGRATION TEST COMPLETED SUCCESSFULLY! 🎉")

    except Exception as e :
        logger .error (f"❌ INTEGRATION TEST FAILED: {e }",exc_info =True )
        sys .exit (1 )


if __name__ =="__main__":


    if "--attention-off"in sys .argv :
        logger .warning ("The '--attention-off' flag is not implemented.")
        logger .warning (
        "Disabling the attention mechanism requires modifying the core model architecture in ThoughtGenerator."
        )
        logger .warning ("Proceeding with attention enabled.")
        sys .argv .remove ("--attention-off")

    main ()
