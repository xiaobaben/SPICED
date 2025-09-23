import time
from utils.util import fix_randomness
import random
import torch
from model.backbone import FeatureExtractor, TransformerEncoder, SleepMLP
from model.backbone import FeatureExtractor_Face, TransformerEncoder_Face, EmotionMLP_Face
from model.backbone import FeatureExtractor_BCI2000, TransformerEncoder_BCI2000, MotorImageryBCI2000

import os
from dataloader.data_loader import Builder, SynapticDatasetBuilder
from torch.utils.data import DataLoader
import numpy as np
from model.incremental_model import SynapticFramework
from utils.util import Evaluator
from utils.util_block import evaluator, weighted_model_fusion


def model_load(args, load_path, block):
    block[0].load_state_dict(torch.load(f"{load_path}/feature_extractor_parameter_{args.seed}.pkl"))
    block[1].load_state_dict(torch.load(f"{load_path}/feature_encoder_parameter_{args.seed}.pkl"))
    block[2].load_state_dict(torch.load(f"{load_path}/sleep_classifier_parameter_{args.seed}.pkl"))
    return block[0], block[1], block[2]


def get_new_task_loader(args, new_task_idx, synaptic_net, is_replay, source_net):
    new_task_path = [[], []]
    file_path = args.file_path + f"/{new_task_idx}/data"
    label_path = args.file_path + f"/{new_task_idx}/label"
    num = 0
    while os.path.exists(file_path + f"/{num}.npy"):
        new_task_path[0].append(file_path + f"/{num}.npy")
        new_task_path[1].append(label_path + f"/{num}.npy")
        num += 1
    if is_replay:
        new_task_builder = SynapticDatasetBuilder(new_task_path, synaptic_net, f"Sub_{new_task_idx}", args, source_net).Dataset
    else:
        new_task_builder = Builder(new_task_path, args).Dataset
    new_task_loader = DataLoader(dataset=new_task_builder, batch_size=args.batch, shuffle=True, num_workers=4)
    return new_task_loader


def trainer(new_task_idx, args, synaptic_net, source_net):
    if args.stability:
        if args.stability == 1:
            fix_randomness(args.seed + 2)
            print(f'Stability Seed: {args.seed + 2}')
            new_task_idx = sorted(new_task_idx, key=lambda x: random.random())
        elif args.stability == 2:
            fix_randomness(args.seed + 3)
            print(f'Stability Seed: {args.seed + 3}')
            new_task_idx = sorted(new_task_idx, key=lambda x: random.random())
        elif args.stability == 3:
            fix_randomness(args.seed + 4)
            print(f'Stability Seed: {args.seed + 4}')
            new_task_idx = sorted(new_task_idx, key=lambda x: random.random())
        elif args.stability == 4:
            fix_randomness(args.seed + 5)
            print(f'Stability Seed: {args.seed + 5}')
            new_task_idx = sorted(new_task_idx, key=lambda x: random.random())
        elif args.stability == 5:
            fix_randomness(args.seed + 6)
            print(f'Stability Seed: {args.seed + 6}')
            new_task_idx = sorted(new_task_idx, key=lambda x: random.random())
        elif args.stability == 6:
            fix_randomness(args.seed + 9)
            print(f'Stability Seed: {args.seed + 9}')
            new_task_idx = sorted(new_task_idx, key=lambda x: random.random())
        elif args.stability == 7:
            fix_randomness(args.seed + 99)
            print(f'Stability Seed: {args.seed + 99}')
            new_task_idx = sorted(new_task_idx, key=lambda x: random.random())
        elif args.stability == 8:
            fix_randomness(args.seed + 999)
            print(f'Stability Seed: {args.seed + 999}')
            new_task_idx = sorted(new_task_idx, key=lambda x: random.random())
    print(new_task_idx)
    results = {i: {"ACC": [], "MF1": []} for i in new_task_idx}
    for num, new_task_id in enumerate(new_task_idx):
        print("Incremental Individual Id", new_task_id)
        if args.dataset == 'ISRUC':
            feature_extractor = FeatureExtractor(args).float()
            classifier = SleepMLP(args).float()
            feature_encoder = TransformerEncoder(args).float()
        elif args.dataset == 'FACED':
            feature_extractor = FeatureExtractor_Face(args).float()
            classifier = EmotionMLP_Face(args).float()
            feature_encoder = TransformerEncoder_Face(args).float()
        elif args.dataset == 'BCI2000':
            feature_extractor = FeatureExtractor_BCI2000(args).float()
            classifier = MotorImageryBCI2000(args).float()
            feature_encoder = TransformerEncoder_BCI2000(args).float()
        else:
            feature_extractor, classifier, feature_encoder = None, None, None

        if not os.path.exists(f"model_parameter/{args.dataset}/{args.save_info}/individual_{num}"):
            os.makedirs(f"model_parameter/{args.dataset}/{args.save_info}/individual_{num}")

        if num == 0:
            """Load initial model """
            load_path = f"model_parameter/{args.dataset}/Pretrain/{args.source_percent}"
            feature_extractor, feature_encoder, classifier = \
                model_load(args, load_path, (feature_extractor, feature_encoder, classifier))
        else:
            """Load last model"""
            load_path = f"model_parameter/{args.dataset}/{args.save_info}/individual_{num-1}"
            feature_extractor, feature_encoder, classifier = \
                model_load(args, load_path, (feature_extractor, feature_encoder, classifier))

        cur_model = (feature_extractor, feature_encoder, classifier)

        start = time.time()
        new_model = incremental_trainer(args, new_task_id, num, synaptic_net, source_net)
        end = time.time()

        args.times.append(end-start)
        """Store Newest Model"""
        state_f = new_model[0].state_dict()
        for key in state_f.keys():
            state_f[key] = state_f[key].to(torch.device("cpu"))

        state_encoder = new_model[1].state_dict()
        for key in state_encoder.keys():
            state_encoder[key] = state_encoder[key].to(torch.device("cpu"))

        state_classifier = new_model[2].state_dict()
        for key in state_classifier.keys():
            state_classifier[key] = state_classifier[key].to(torch.device("cpu"))

        save_f = f"model_parameter/{args.dataset}/{args.save_info}/" \
                 f"individual_{num}/feature_extractor_parameter_{args.seed}.pkl"
        save_e = f"model_parameter/{args.dataset}/{args.save_info}/" \
                 f"individual_{num}/feature_encoder_parameter_{args.seed}.pkl"
        save_c = f"model_parameter/{args.dataset}/{args.save_info}/" \
                 f"individual_{num}/sleep_classifier_parameter_{args.seed}.pkl"
        torch.save(state_f, save_f)
        torch.save(state_encoder, save_e)
        torch.save(state_classifier, save_c)

        # Update Synaptic Node Information
        synaptic_net.nodes[f"Sub_{new_task_id}"].model_path.append(save_f)
        synaptic_net.nodes[f"Sub_{new_task_id}"].model_path.append(save_e)
        synaptic_net.nodes[f"Sub_{new_task_id}"].model_path.append(save_c)

        """Initial Model"""
        if args.dataset == 'ISRUC':
            feature_extractor_initial = FeatureExtractor(args).float()
            classifier_initial = SleepMLP(args).float()
            feature_encoder_initial = TransformerEncoder(args).float()
        elif args.dataset == 'FACED':
            feature_extractor_initial = FeatureExtractor_Face(args).float()
            classifier_initial = EmotionMLP_Face(args).float()
            feature_encoder_initial = TransformerEncoder_Face(args).float()
        elif args.dataset == 'BCI2000':
            feature_extractor_initial = FeatureExtractor_BCI2000(args).float()
            classifier_initial = MotorImageryBCI2000(args).float()
            feature_encoder_initial = TransformerEncoder_BCI2000(args).float()
        else:
            feature_extractor_initial, classifier_initial, feature_encoder_initial = None, None, None
        load_path = f"model_parameter/{args.dataset}/Pretrain/{args.source_percent}"
        feature_extractor_initial, feature_encoder_initial, classifier_initial = \
            model_load(args, load_path, (feature_extractor_initial, feature_encoder_initial, classifier_initial))

        """Metric"""
        new_task_loader = get_new_task_loader(args, new_task_id, synaptic_net, is_replay=False, source_net=source_net)
        new_task_initial_ans = evaluator((feature_extractor_initial,
                                          feature_encoder_initial,
                                          classifier_initial), new_task_loader, args)

        new_task_before_ans = evaluator(cur_model, new_task_loader, args)
        new_task_after_ans = evaluator(new_model, new_task_loader, args)

        new_initial_evaluator = Evaluator(new_task_initial_ans[0], new_task_initial_ans[1])
        new_before_evaluator = Evaluator(new_task_before_ans[0], new_task_before_ans[1])
        new_after_evaluator = Evaluator(new_task_after_ans[0], new_task_after_ans[1])

        new_task_initial_acc, new_task_initial_mf1 = new_initial_evaluator.metric_acc(), new_initial_evaluator.metric_mf1(args.dataset)
        new_task_before_acc, new_task_before_mf1 = new_before_evaluator.metric_acc(), new_before_evaluator.metric_mf1(args.dataset)
        new_task_after_acc, new_task_after_mf1 = new_after_evaluator.metric_acc(), new_after_evaluator.metric_mf1(args.dataset)

        results[new_task_id]['ACC'] = [new_task_initial_acc, new_task_before_acc, new_task_after_acc]
        results[new_task_id]['MF1'] = [new_task_initial_mf1, new_task_before_mf1, new_task_after_mf1]

        print(f"=========Incremental Individual {new_task_id}=========")
        print(" ACC Initial                    ", results[new_task_id]['ACC'][0], "\n",
              "ACC Before Continual Learning   ", results[new_task_id]['ACC'][1], "\n",
              "ACC After Continual Learning    ", results[new_task_id]['ACC'][2], "\n"
              " MF1 Initial                    ",  results[new_task_id]['MF1'][0], "\n",
              "MF1 Before Continual Learning   ", results[new_task_id]['MF1'][1], "\n",
              "MF1 After Continual Learning    ", results[new_task_id]['MF1'][2], "\n"
              )

        saving_new_node_sample(args, new_task_id, num, cur_model, synaptic_net, source_net)
        synaptic_net.global_renormalization_synapses()
    return results, synaptic_net


def saving_new_node_sample(args, new_task_id, num, model, synaptic_net, source_net):
    new_task_path = [[], []]
    file_path = args.file_path + f"/{new_task_id}/data"
    label_path = args.file_path + f"/{new_task_id}/label"
    idx = 0
    while os.path.exists(file_path + f"/{idx}.npy"):
        new_task_path[0].append(file_path + f"/{idx}.npy")
        new_task_path[1].append(label_path + f"/{idx}.npy")
        idx += 1
    new_task_loader = get_new_task_loader(args, new_task_id, synaptic_net, is_replay=False, source_net=source_net)
    new_task_after_ans = evaluator(model, new_task_loader, args)

    save_path = f"model_parameter/{args.dataset}/{args.save_info}/individual_{num}/label"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    confidence_level = args.saving_confidence
    new_task_out = new_task_after_ans[2]
    mean_t_pred = torch.softmax(new_task_out, dim=1)
    pred_prob = mean_t_pred.max(1, keepdim=True)[0].squeeze()
    pred_label = mean_t_pred.max(1, keepdim=True)[1].squeeze()
    pred_prob = pred_prob.cpu().numpy()
    pred_label = pred_label.cpu().numpy()

    if args.dataset == 'ISRUC':
        confident_epoch_n = 15
        for bh in range(pred_prob.shape[0]):
            confident_epoch_num_per_seq = np.sum(pred_prob[bh, :] >= confidence_level)
            if confident_epoch_num_per_seq >= confident_epoch_n:
                confident_label = pred_label[bh, :].reshape(-1, 1)
                confident_label = np.squeeze(confident_label)
                save_label_path = save_path + f"/{bh}.npy"
                np.save(save_label_path, confident_label)
                # Update Synaptic Node pseudo-label information
                synaptic_net.nodes[f'Sub_{new_task_id}'].sample_path['data'].append(new_task_path[0][bh])
                synaptic_net.nodes[f'Sub_{new_task_id}'].sample_path['label'].append(save_label_path)
    if args.dataset in ['FACED', 'BCI2000']:
        for bh in range(pred_prob.shape[0]):
            if pred_prob[bh] >= confidence_level:
                confident_label = pred_label[bh].reshape(-1, 1)
                confident_label = np.squeeze(confident_label)
                save_label_path = save_path + f"/{bh}.npy"
                np.save(save_label_path, confident_label)
                # Update Synaptic Node pseudo-label information
                synaptic_net.nodes[f'Sub_{new_task_id}'].sample_path['data'].append(new_task_path[0][bh])
                synaptic_net.nodes[f'Sub_{new_task_id}'].sample_path['label'].append(save_label_path)

    print(synaptic_net.nodes[f'Sub_{new_task_id}'].name,
          " saving data num:", len(synaptic_net.nodes[f'Sub_{new_task_id}'].sample_path['data']))


def incremental_trainer(args, new_task_id, num, synaptic_net, source_net):
    """Add new individual to synaptic network"""
    prototype_path = args.prototype_path + f"/{new_task_id}.npy"
    prototype = np.load(prototype_path)
    synaptic_net.add_node(f"Sub_{new_task_id}", prototype, args, similarity_threshold=args.similarity_threshold)

    """Determine whether there is a connection with existing nodes"""
    collections_num = synaptic_net.nodes[f'Sub_{new_task_id}'].connection_count()
    is_exist_connection = True if collections_num >= 1 else False
    new_task_loader = get_new_task_loader(args, new_task_id, synaptic_net, is_replay=is_exist_connection, source_net=source_net)

    if args.dataset == 'ISRUC':
        feature_extractor = FeatureExtractor(args).float()
        classifier = SleepMLP(args).float()
        feature_encoder = TransformerEncoder(args).float()
    elif args.dataset == 'FACED':
        feature_extractor = FeatureExtractor_Face(args).float()
        classifier = EmotionMLP_Face(args).float()
        feature_encoder = TransformerEncoder_Face(args).float()
    elif args.dataset == 'BCI2000':
        feature_extractor = FeatureExtractor_BCI2000(args).float()
        classifier = MotorImageryBCI2000(args).float()
        feature_encoder = TransformerEncoder_BCI2000(args).float()
    else:
        feature_extractor, classifier, feature_encoder = None, None, None

    backbone = []
    """Model Fusion"""
    if not is_exist_connection:
        """If no connection exists, use the fused model from the three most similar nodes"""
        max_sim = -999
        exist_sub = None
        model_path = f"model_parameter/{args.dataset}/Pretrain/{args.source_percent}"  # initial model
        for exist in list(synaptic_net.nodes.values()):
            if exist.name != f"Sub_{new_task_id}":
                sim = synaptic_net.sim_calculator.domain_weighted_cosine(prototype, exist.prototype)
                if sim > max_sim:
                    max_sim = sim
                    model_path = exist.model_path
                    exist_sub = exist.name
        last_slash_index = model_path[0].rfind('/')
        model_path = model_path[0][:last_slash_index]
        feature_extractor, feature_encoder, sleep_classifier = \
            model_load(args, model_path, (feature_extractor, feature_encoder, classifier))
        backbone = [feature_extractor, feature_encoder, sleep_classifier]
        print("Current Incremental Individual do not exist collections")
        print(f"The most similar individual is {exist_sub}")
    else:
        """Fuse the models corresponding to the N nodes with the highest importance"""
        importance = synaptic_net.nodes[f"Sub_{new_task_id}"].find_importance_knn(args)
        for i, module in enumerate([feature_extractor, feature_encoder, classifier]):
            fusion_list = []
            importance_weights = []
            for fusion_x in importance:
                pth = synaptic_net.nodes[f"{fusion_x[0]}"].model_path[i]
                new_module = type(module)(args).float()
                new_module.load_state_dict(torch.load(pth))
                fusion_list.append(new_module)
                importance_weights.append(fusion_x[1])
            backbone.append(weighted_model_fusion(fusion_list, importance_weights, args))
        print(f"Current Incremental Individual exists {collections_num} collections")
        print(f"The most importance Nodes are: {importance}")

    framework = SynapticFramework2(backbone, args)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    backbone[0].to(device)
    backbone[1].to(device)
    backbone[2].to(device)

    return_model = []
    return_loss = []
    for epoch in range(1, args.incremental_epoch + 1):
        backbone[0].train()
        backbone[1].train()
        backbone[2].train()
        epoch_loss = []

        for batch_idx, data in enumerate(new_task_loader):
            x_new, x_replay, label = data[0].to(device), data[1].to(device), data[2].to(device)
            loss, model = framework.update(x_new, x_replay, label, is_exist_connection, epoch)
            epoch_loss.append(loss)
        print(f"Incremental Individual ID {int(num)}  Training Epoch {epoch} Loss {np.mean(epoch_loss)}")
        if epoch > 10:
            return_loss.append(np.mean(epoch_loss))
            return_model.append(model)
    print('best selection:', return_loss.index(min(return_loss))+11)
    return return_model[return_loss.index(min(return_loss))]




