import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.model_mil import MIL_fc, MIL_fc_mc
from models.model_clam import CLAM_SB, CLAM_MB
import pdb
import os
import pandas as pd
from utils.utils import *
from utils.core_utils import Accuracy_Logger, forward_bag_model
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import torch.nn as nn
from Model.MambaMIL import MambaMIL
from Model.ABMIL import DAttention
from Model.TransMIL import TransMIL
from Model.Pure_mamba_mil import PureMambaMIL
from Model.MambaProto import ProtoMambaMIL
from Model.ACMIL import ACMIL_GA
from Model.SRMambaProV import SRMambaProMIL
from Model.CARETransMIL import CARETransMIL
from Model.HAFED_CARE import HAFED_CARE_MODEL_TYPES, build_hafed_care_model

def initiate_model(args, ckpt_path, proto=None, device='cuda'):
    print('Init Model')    
    model_dict = {"dropout": args.drop_out, 'n_classes': args.n_classes, "embed_dim": args.embed_dim}
    
    if args.model_size is not None and args.model_type in ['clam_sb', 'clam_mb']:
        model_dict.update({"size_arg": args.model_size})
    
    if args.model_type =='clam_sb':
        model = CLAM_SB(**model_dict)
    elif args.model_type =='clam_mb':
        model = CLAM_MB(**model_dict)
    elif args.model_type == 'mamba_mil':
        args.mambamil_layer = 2
        args.mambamil_rate = 5
        args.mambamil_type = 'Mamba'
        args.in_dim = 1024
        args.n_classes = 2
        model = MambaMIL(num_cluster=1, num_head=1, hidden_size=256,embed_size=512,in_dim=args.in_dim, n_classes=args.n_classes, 
                                attn_dropout=0.1,output_class=2,dropout=0.25, act='gelu',
                                 layer=args.mambamil_layer, rate=args.mambamil_rate, type=args.mambamil_type,init_query=False,
                                query_is_parameter=False, alpha = 0.1, use_inst_proto = False, protoloss = nn.CrossEntropyLoss(), proto=None)
    elif args.model_type == 'mlla_mil':
        from Model.MLLAMIL import MllaMIL
        args.in_dim = 1024
        args.n_classes = 2
        model = MllaMIL(dim=args.in_dim, n_classes=args.n_classes,num_heads=4, mlp_ratio=2., qkv_bias=True, drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, use_inst_proto=True, protoloss = nn.CrossEntropyLoss(),num_cluster=2,
                 num_head=1,hidden_size=256, embed_size=512, init_query=False,
                 query_is_parameter=False, attn_dropout=0.1, dropout=0.25)

    elif args.model_type == 'trans_mil':
        from Model.TransMIL import TransMIL
        args.in_dim = 1024
        args.n_classes = 2
        args.drop_out=0.25
        model = TransMIL(args.in_dim, args.n_classes, dropout = args.drop_out, act='relu', survival = False)

    elif args.model_type == 'att_mil':
        from Model.ABMIL import DAttention
        args.in_dim = 1024
        args.n_classes = 2
        args.drop_out=0.25
        model = DAttention(args.in_dim, args.n_classes, dropout=args.drop_out, act='relu', survival=False)

    elif args.model_type == 'pure_mamba_mil':
        from Model.Pure_mamba_mil import PureMambaMIL
        args.in_dim = 1024
        args.n_classes = 2
        args.mambamil_layer = 2
        args.mambamil_rate = 5
        args.mambamil_type = 'Mamba'
        model = PureMambaMIL(in_dim=args.in_dim, n_classes=args.n_classes,dropout=0.25, act='gelu',
                                layer=args.mambamil_layer, rate=args.mambamil_rate, type=args.mambamil_type)

    elif args.model_type == 'pmil':
        from Model.ProMIL import ProtoMIL
        model = ProtoMIL(instance_eval = False, similarity_method="Cosine", aggregation_method='weightedsum_prototype')

    elif args.model_type == 'protomamba':
        from Model.MambaProto import ProtoMambaMIL
        model = ProtoMambaMIL(num_cluster=1, num_head=1, in_dim=1024, n_classes=2, attn_dropout=0.1, dropout=0.25, act='gelu', init_query=False,
                 query_is_parameter=False, layer=2, rate=10, type="Mamba", use_constrast = False, use_inst_proto = False, protoloss = nn.CrossEntropyLoss())

    elif args.model_type == 'acmil':
        from Model.ACMIL import ACMIL_GA
        model = ACMIL_GA(n_masked_patch=50,n_token=6, mask_drop=0.6)

    if args.model_type == 'mhamil':
        from Model.MHAMIL import SUBMHAMIL
        model = SUBMHAMIL(n_masked_patch=10,n_token=5, mask_drop=0.6)

    if args.model_type == 'mhapromil':
        from Model.MHAProMIL import MHAPROMIL
        model = MHAPROMIL(n_masked_patch=10,n_token=6, mask_drop=0.6,prototype_vector=proto)
    if args.model_type == 'mean_mil':
        from Model.Mean_Max_MIL import MeanMIL
        args.in_dim = 1024
        args.n_classes = 2
        model = MeanMIL(args.in_dim, args.n_classes, survival=False)
    if args.model_type == 'max_mil':
        from Model.Mean_Max_MIL import MaxMIL
        args.in_dim = 1024
        args.n_classes = 2
        model = MaxMIL(args.in_dim, args.n_classes, survival=False)
    if args.model_type == 'care_trans_mil':
        model = CARETransMIL(
            in_dim=args.in_dim,
            n_classes=args.n_classes,
            dropout=args.care_dropout,
            act=args.care_act,
            survival=False,
            embed_dim=args.care_embed_dim,
            depths=args.care_depths,
            drop_path_rate=args.care_drop_path_rate,
        )
    # ===== NEW CODE START: evaluate integrated HAFED =====
    if args.model_type == 'hafed':
        from Model.HAFED import HAFEDMIL
        model = HAFEDMIL(
            in_dim=args.in_dim,
            n_classes=args.n_classes,
            inner_dim=args.hafed_inner_dim,
            attn_dim=args.hafed_attn_dim,
            dropout=args.drop_out,
            n_token_1=args.hafed_n_token_1,
            n_token_2=args.hafed_n_token_2,
            n_masked_patch_1=args.hafed_n_masked_patch_1,
            n_masked_patch_2=args.hafed_n_masked_patch_2,
            mask_drop=args.hafed_mask_drop,
            dim_reduction=args.hafed_dim_reduction,
        )
    if args.model_type in HAFED_CARE_MODEL_TYPES:
        model = build_hafed_care_model(
            args.model_type,
            in_dim=args.in_dim,
            n_classes=args.n_classes,
            inner_dim=args.hafed_inner_dim,
            attn_dim=args.hafed_attn_dim,
            dropout=args.drop_out,
            n_token_1=args.hafed_n_token_1,
            n_token_2=args.hafed_n_token_2,
            n_masked_patch_1=args.hafed_n_masked_patch_1,
            n_masked_patch_2=args.hafed_n_masked_patch_2,
            mask_drop=args.hafed_mask_drop,
            dim_reduction=args.hafed_dim_reduction,
            care_embed_dim=args.care_embed_dim,
            care_depths=args.care_depths,
            care_drop_path_rate=args.care_drop_path_rate,
            care_act=args.care_act,
            care_dropout=args.care_dropout,
        )
    # ===== NEW CODE END: evaluate integrated HAFED =====

    if args.model_type == 'phiher2':
        from Model.PHIHERMIL import PhiHER2model
        model = PhiHER2model(feature_size=1024, embed_size=512, hidden_size=128, num_head=1, num_cluster=6, inst_num=None, 
                                inst_num_twice=500,random_inst=False,attn_dropout=0., dropout=0.25, output_class=2, 
                                cls_method="cls_keep_prototype_dim",  abmil_branch=True, init_query=False, query_is_parameter=False,
                                only_similarity=True)

    if args.model_type == 'srmambapromil':
            from Model.SRMambaProV import SRMambaProMIL
            model = SRMambaProMIL(in_dim=1024, n_masked_patch=10, n_token=6, mask_drop=0.6, prototype_vector=proto)  

    # else: # args.model_type == 'mil'
    #     if args.n_classes > 2:
    #         model = MIL_fc_mc(**model_dict)
    #     else:
    #         model = MIL_fc(**model_dict)

    print_network(model)

    ckpt = torch.load(ckpt_path)
    ckpt_clean = {}
    for key in ckpt.keys():
        if 'instance_loss_fn' in key:
            continue
        ckpt_clean.update({key.replace('.module', ''):ckpt[key]})
    model.load_state_dict(ckpt_clean, strict=True)

    _ = model.to(device)
    _ = model.eval()
    return model

def eval(dataset, args, ckpt_path):
    model = initiate_model(args, ckpt_path)
    
    print('Init Loaders')
    loader = get_simple_loader(dataset)
    patient_results, test_error, auc, df, _ = summary(model, loader, args)
    print('test_error: ', test_error)
    print('auc: ', auc)
    return model, patient_results, test_error, auc, df

def summary(model, loader, args):
    acc_logger = Accuracy_Logger(n_classes=args.n_classes)
    model.eval()
    test_loss = 0.
    test_error = 0.

    all_probs = np.zeros((len(loader), args.n_classes))
    all_labels = np.zeros(len(loader))
    all_preds = np.zeros(len(loader))

    slide_ids = loader.dataset.slide_data['slide_id']
    patient_results = {}
    for batch_idx, (data, label) in enumerate(loader):
        data, label = data.to(device), label.to(device)
        slide_id = slide_ids.iloc[batch_idx]
        with torch.no_grad():
            _, logits, Y_prob, Y_hat, _, results_dict = forward_bag_model(model, data, args)
        
        acc_logger.log(Y_hat, label)
        
        probs = Y_prob.squeeze(0).cpu().numpy()

        all_probs[batch_idx] = probs
        all_labels[batch_idx] = label.item()
        all_preds[batch_idx] = Y_hat.item()
        
        patient_results.update({slide_id: {'slide_id': np.array(slide_id), 'prob': probs, 'label': label.item()}})
        
        error = calculate_error(Y_hat, label)
        test_error += error

    del data
    test_error /= len(loader)

    aucs = []
    if len(np.unique(all_labels)) == 1:
        auc_score = -1

    else: 
        if args.n_classes == 2:
            auc_score = roc_auc_score(all_labels, all_probs[:, 1])
        else:
            binary_labels = label_binarize(all_labels, classes=[i for i in range(args.n_classes)])
            for class_idx in range(args.n_classes):
                if class_idx in all_labels:
                    fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], all_probs[:, class_idx])
                    aucs.append(auc(fpr, tpr))
                else:
                    aucs.append(float('nan'))
            if args.micro_average:
                binary_labels = label_binarize(all_labels, classes=[i for i in range(args.n_classes)])
                fpr, tpr, _ = roc_curve(binary_labels.ravel(), all_probs.ravel())
                auc_score = auc(fpr, tpr)
            else:
                auc_score = np.nanmean(np.array(aucs))

    results_dict = {'slide_id': slide_ids, 'Y': all_labels, 'Y_hat': all_preds}
    for c in range(args.n_classes):
        results_dict.update({'p_{}'.format(c): all_probs[:,c]})
    df = pd.DataFrame(results_dict)
    return patient_results, test_error, auc_score, df, acc_logger
