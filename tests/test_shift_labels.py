#!/usr/bin/env python3

import importlib .util 
import pathlib 
import torch 

module_path =pathlib .Path ('mgm_project/scripts/6-10-25/train_geometric_model_v2.py')
spec =importlib .util .spec_from_file_location ('train_geometric_model_v2',module_path )
tgm =importlib .util .module_from_spec (spec )
spec .loader .exec_module (tgm )


def test_safe_collate_fn_shifted_labels ():
    batch =[
    {'input_ids':torch .tensor ([1 ,2 ,3 ,4 ]),
    'attention_mask':torch .tensor ([1 ,1 ,1 ,1 ])},
    {'input_ids':torch .tensor ([5 ,6 ,7 ,8 ]),
    'attention_mask':torch .tensor ([1 ,1 ,1 ,1 ])},
    ]
    out =tgm .safe_collate_fn (batch )
    expected =torch .tensor ([[2 ,3 ,4 ,-100 ],[6 ,7 ,8 ,-100 ]])
    assert torch .equal (out ['labels'],expected )


def test_collate_fn_streaming_shifted_labels ():
    batch =[
    {'input_ids':torch .tensor ([1 ,2 ,3 ,4 ])},
    {'input_ids':torch .tensor ([5 ,6 ])},
    ]
    out =tgm .collate_fn_streaming (batch )

    assert torch .equal (out ['labels'][0 ,:4 ],torch .tensor ([2 ,3 ,4 ,-100 ]))

    assert torch .equal (out ['labels'][1 ,:2 ],torch .tensor ([6 ,-100 ]))

