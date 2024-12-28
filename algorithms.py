"""
This code collected some methods from DomainBed (https://github.com/facebookresearch/DomainBed) and other SOTA methods.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os, collections, logging
from collections import OrderedDict
from info_nce import InfoNCE
from thop import profile

import utils.misc as misc
from utils.validate import algorithm_validate, algorithm_validate_iw, algorithm_eval_iw, algorithm_eval_mu, algorithm_eval_tsne, algorithm_eval_heat
import modeling.model_manager as models
from modeling.losses import DahLoss
from modeling.nets import LossValley, AveragedModel
from dataset.data_manager import get_post_FundusAug

from backpack import backpack, extend
from backpack.extensions import BatchGrad

ALGORITHMS = [
    'ERM',
    'GDRNet',
    'GREEN',
    'CABNet',
    'MixupNet',
    'MixStyleNet',
    'Fishr',
    'DRGen'
    ]

def get_algorithm_class(algorithm_name):
    """Return the algorithm class with the given name."""
    if algorithm_name not in globals():
        raise NotImplementedError("Algorithm not found: {}".format(algorithm_name))
    return globals()[algorithm_name]

class Algorithm(torch.nn.Module):
    """
    A subclass of Algorithm implements a domain generalization algorithm.
    Subclasses should implement the following:
    - update()
    - validate()
    - save_model()
    - renew_model()
    - predict()
    """
    def __init__(self, num_classes, cfg):
        super(Algorithm, self).__init__()
        self.cfg = cfg
        self.epoch = 0

    def update(self, minibatches):
        raise NotImplementedError
    
    def update_epoch(self, epoch):
        self.epoch = epoch
        return epoch
    
    def validate(self, val_loader, test_loader, writer):
        raise NotImplementedError
    
    def save_model(self, log_path):
        raise NotImplementedError
    
    def renew_model(self, log_path):
        raise NotImplementedError
    
    def predict(self, x):
        raise NotImplementedError

class ERM(Algorithm):
    """Empirical Risk Minimization (ERM)"""
    def __init__(self, num_classes, cfg):
        super(ERM, self).__init__(num_classes, cfg)
        
        self.network = models.get_net(cfg)

        # input = torch.randn(16, 3, 224, 224)
        # flops, params = profile(self.network, inputs=(input, ))
        # print('FLOPs = ' + str(flops/1000**3) + 'G')
        # print('Params = ' + str(params/1000**2) + 'M')
        # exit()

        if cfg.BACKBONE == 'swint' or cfg.BACKBONE == 'swint_iw':
            self.classifier = models.get_classifier(1024, cfg)
        else:
            self.classifier = models.get_classifier(self.network.out_features(), cfg)

        self.optimizer = torch.optim.SGD(
            [{"params":self.network.parameters()},
            {"params":self.classifier.parameters()}],
            lr = cfg.LEARNING_RATE,
            momentum = cfg.MOMENTUM,
            weight_decay = cfg.WEIGHT_DECAY,
            nesterov=True)

    def update(self, minibatch):
        image, mask, label, domain = minibatch
        self.optimizer.zero_grad()

        features = self.network(image)
        output = self.classifier(features)
        loss = F.cross_entropy(output, label)
        
        loss.backward()
        self.optimizer.step()

        return {'loss':loss}
    
    # def validate(self, val_loader, test_loader, writer):
    #     val_auc = -1
    #     test_auc = -1
    #     if self.epoch <= self.cfg.EPOCHS:
    #         if self.cfg.BACKBONE == 'swint_iw':
    #             val_auc, val_loss = algorithm_validate_iw(self, val_loader, writer, self.epoch, 'val')
    #             test_auc, test_loss = algorithm_validate_iw(self, test_loader, writer, self.epoch, 'test')
    #         else:
    #             val_auc, val_loss = algorithm_validate(self, val_loader, writer, self.epoch, 'val')
    #             test_auc, test_loss = algorithm_validate(self, test_loader, writer, self.epoch, 'test')
    #         if self.epoch == self.cfg.EPOCHS:
    #             self.epoch += 1
    #     else:
    #         if self.cfg.BACKBONE == 'swint_iw':
    #             test_auc, test_loss = algorithm_validate_iw(self, test_loader, writer, self.cfg.EPOCHS + self.cfg.VAL_EPOCH, 'test')
    #         else:
    #             test_auc, test_loss = algorithm_validate(self, test_loader, writer, self.cfg.EPOCHS + self.cfg.VAL_EPOCH, 'test')
    #         logging.info('Best performance on test domain(s): {}'.format(test_auc))
                
    #     return val_auc, test_auc
    def validate(self, val_loader, test_loader, writer):
        val_acc = -1
        test_acc = -1
        if self.epoch <= self.cfg.EPOCHS:
            if self.cfg.BACKBONE == 'swint_iw':
                val_acc, val_loss = algorithm_validate_iw(self, val_loader, writer, self.epoch, 'val')
                test_acc, test_loss = algorithm_validate_iw(self, test_loader, writer, self.epoch, 'test')
            else:
                val_acc, val_loss = algorithm_validate(self, val_loader, writer, self.epoch, 'val')
                test_acc, test_loss = algorithm_validate(self, test_loader, writer, self.epoch, 'test')
            if self.epoch == self.cfg.EPOCHS:
                self.epoch += 1
        else:
            if self.cfg.BACKBONE == 'swint_iw':
                test_acc, test_loss = algorithm_validate_iw(self, test_loader, writer, self.cfg.EPOCHS + self.cfg.VAL_EPOCH, 'test')
            else:
                test_acc, test_loss = algorithm_validate(self, test_loader, writer, self.cfg.EPOCHS + self.cfg.VAL_EPOCH, 'test')
            logging.info('Last performance on test domain(s): {}'.format(test_acc))
                
        return val_acc, test_acc
    
    
    def save_model(self, log_path):
        logging.info("Saving best model...")
        torch.save(self.network.state_dict(), os.path.join(log_path, 'best_model.pth'))
        torch.save(self.classifier.state_dict(), os.path.join(log_path, 'best_classifier.pth'))
        # torch.save(self.network.state_dict(), os.path.join(log_path, 'model_e{}_{:.2f}.pth'.format(epoch, test_acc*100)))
        # torch.save(self.classifier.state_dict(), os.path.join(log_path, 'classifier_e{}_{:.2f}.pth'.format(epoch, test_acc*100)))
    
    def renew_model(self, log_path):
        net_path = os.path.join(log_path, 'best_model.pth')
        classifier_path = os.path.join(log_path, 'best_classifier.pth')
        self.network.load_state_dict(torch.load(net_path))
        self.classifier.load_state_dict(torch.load(classifier_path))

    def predict(self, x):
        if self.cfg.BACKBONE == 'swint_iw':
            # feature, iw_loss, sw_loss = self.network(x)
            # return self.classifier(feature), iw_loss, sw_loss
            feature, iw_loss, l4 = self.network(x)
            return self.classifier(feature), iw_loss, l4
        else:
            return self.classifier(self.network(x))
    
# Our method
class GDRNet(ERM):
    def __init__(self, num_classes, cfg):
        super(GDRNet, self).__init__(num_classes, cfg)
        self.fundusAug = get_post_FundusAug(cfg)
        self.criterion = DahLoss(beta= cfg.GDRNET.BETA, max_iteration = cfg.EPOCHS, \
                                training_domains = cfg.DATASET.SOURCE_DOMAINS, temperature = cfg.GDRNET.TEMPERATURE, \
                                scaling_factor = cfg.GDRNET.SCALING_FACTOR)
        ##### ours
        # self.c_mlp = nn.Linear(1024, 1024)
        # self.n_mlp = nn.Linear(1024, 1024)
        self.c_w = nn.Linear(1024, num_classes, bias=False)
        self.num_classes = num_classes
        ##### ours
                                    
    def img_process(self, img_tensor, mask_tensor, fundusAug):
        
        img_tensor_new, mask_tensor_new = fundusAug['post_aug1'](img_tensor.clone(), mask_tensor.clone())
        img_tensor_new = img_tensor_new * mask_tensor_new
        img_tensor_new = fundusAug['post_aug2'](img_tensor_new)
        img_tensor_ori = fundusAug['post_aug2'](img_tensor)

        return img_tensor_new, img_tensor_ori
    
    def instance_whitening_loss(self, x):
        ### instance whitening
        eps = 1e-5
        f_map = x
        f_map = torch.nn.InstanceNorm2d(f_map.shape[1], affine=False)(f_map)
        B, C, H, W = f_map.shape  # i-th feature size (B X C X H X W)
        HW = H * W
        eye = torch.eye(C).cuda()
        f_map = f_map.contiguous().view(B, C, -1)  # B X C X H X W > B X C X (H X W)
        f_cor = torch.bmm(f_map, f_map.transpose(1, 2)).div(HW-1) + (eps * eye)  # C X C / HW
        f_cor_masked = torch.triu(f_cor) # take upper triangular matrix

        off_diag_sum = torch.sum(torch.abs(f_cor_masked), dim=(1,2), keepdim=True) - 0 # B X 1 X 1
        loss = torch.clamp(torch.div(off_diag_sum, 1), min=0) # B X 1 X 1
        loss = torch.sum(loss) / B
        return loss
    
    def infonce_distance(self, x1, x2):
        # Custom Distance Function
        infonce_loss = InfoNCE()
        return infonce_loss(x1, x2)
    
    def update(self, minibatch, margin):
        
        image, mask, label, domain = minibatch
        
        self.optimizer.zero_grad()
        
        image_new, image_ori = self.img_process(image, mask, self.fundusAug)
        
        if self.cfg.BACKBONE == 'swint_iw':
            # features_ori, iw_loss_ori, sw_loss_ori, l4_ori = self.network(image_ori)
            # features_new, iw_loss_new, sw_loss_new, l4_new = self.network(image_new)
            features_ori, iw_loss_ori, l4_ori = self.network(image_ori)
            features_new, iw_loss_new, l4_new = self.network(image_new)
        else:
            features_ori = self.network(image_ori)
            features_new = self.network(image_new)
        
        # ##### ours
        # label_ = torch.eye(self.num_classes)[label,:]
        # f_c = self.c_mlp(features_new)
        # f_n = self.n_mlp(features_new)
        # T_c = torch.mul(self.c_w(f_c), label_.cuda())
        # MI_c = torch.abs(torch.mean(T_c, dim=0) - torch.log(torch.mean(torch.mean(torch.exp(T_c), dim=1), dim=0)))
        # MI_n = torch.abs(torch.mean(torch.log(torch.nn.Softmax()(f_n)), dim=0) - torch.mean(torch.mean(torch.log(torch.nn.Softmax()(f_n)), dim=1), dim=0))
        # MI_c = torch.mean(MI_c)
        # MI_n = torch.mean(MI_n)
        # B, C = f_c.shape[:]
        # f_c_ = f_c.reshape(B, 1, 1, C)
        # f_c_ = nn.InstanceNorm2d(C)(f_c_)
        # f_c = f_c_.reshape(B, C)
        # # ciw_loss = self.instance_whitening_loss(f_c_)
        # output_new = self.classifier((f_c + f_n))
        # ##### ours

        ##### ours triplet loss
        loss_trip_ori = iw_loss_ori - iw_loss_ori
        loss_trip_new = iw_loss_ori - iw_loss_ori
        B, N, C = l4_ori.shape[:]
        triplet_loss = nn.TripletMarginLoss(margin=margin)
        for a_i in range(B):
            positive_indexs = (label == label[a_i]).nonzero()
            negative_indexs = (label != label[a_i]).nonzero()
            if len(positive_indexs) >= 2 and len(negative_indexs) >= 1:
                anchor_index = a_i
                positive_index = (label == label[a_i]).nonzero()[1].item()
                negative_index = (label != label[a_i]).nonzero()[0].item()
                anchor_ori = l4_ori[anchor_index].squeeze()
                positive_ori = l4_ori[positive_index].squeeze()
                negative_ori = l4_ori[negative_index].squeeze()
                anchor_new = l4_new[anchor_index].squeeze()
                positive_new = l4_new[positive_index].squeeze()
                negative_new = l4_new[negative_index].squeeze()
                triplet_loss = nn.TripletMarginWithDistanceLoss(distance_function=self.infonce_distance, margin=1.0)
                loss_trip_ori = triplet_loss(anchor_ori, positive_ori, negative_ori)
                loss_trip_new = triplet_loss(anchor_new, positive_new, negative_new)
            else:
                continue

        ##### ours triplet loss

        output_new = self.classifier(features_new)

        loss, loss_dict_iter = self.criterion([output_new], [features_ori, features_new], label, domain)
        
        if self.cfg.BACKBONE == 'swint_iw':
            # print(loss)
            # print((iw_loss_ori + iw_loss_new) * 1e-6)
            # print((sw_loss_ori + sw_loss_new))
            # exit()
            # loss += (iw_loss_ori + iw_loss_new) * 1e-6 + (sw_loss_ori + sw_loss_new)
            # print(loss)
            # print((iw_loss_ori + iw_loss_new) * 1e-6)
            # print(loss_trip_ori + loss_trip_new)
            # exit()
            loss += (iw_loss_ori + iw_loss_new) * 1e-6 + (loss_trip_ori + loss_trip_new) * 1e-2
            # loss += (MI_c + MI_n) * 0.1
        
        loss.backward()
        self.optimizer.step()

        return loss_dict_iter
    
    def update_epoch(self, epoch):
        self.epoch = epoch
        return self.criterion.update_alpha(epoch)
    
    def load_model(self, model_path):
        checkpoint = torch.load(os.path.join(model_path, 'best_model.pth'))
        self.network.load_state_dict(checkpoint)
        checkpoint_ = torch.load(os.path.join(model_path, 'best_classifier.pth'))
        self.classifier.load_state_dict(checkpoint_)
    
    def eval_cls(self, val_loader, test_loader, writer):
        test_acc = -1
        test_acc = algorithm_eval_iw(self, test_loader, writer, self.cfg.EPOCHS + self.cfg.VAL_EPOCH, 'test')
        logging.info('Best performance on test domain(s): {}'.format(test_acc))
        return test_acc

    def eval_mu(self, val_loader, test_loader):
        mu, sigma = algorithm_eval_mu(self, test_loader, self.cfg.EPOCHS + self.cfg.VAL_EPOCH, 'test')
        return mu, sigma
    
    def eval_tsne(self, val_loader, test_loader):
        feats, labels = algorithm_eval_tsne(self, test_loader, self.cfg.EPOCHS + self.cfg.VAL_EPOCH, 'test')
        return feats, labels
    
    def eval_heat(self, val_loader, test_loader):
        feats = algorithm_eval_heat(self, test_loader, self.cfg.EPOCHS + self.cfg.VAL_EPOCH, 'test')
        return feats

class GREEN(Algorithm):
    def __init__(self, num_classes, cfg):
        super(GREEN, self).__init__(num_classes, cfg)
        self.network = models.get_net(cfg)
        self.optimizer = torch.optim.SGD(
            self.network.parameters(),
            lr = cfg.LEARNING_RATE,
            momentum = cfg.MOMENTUM,
            weight_decay = cfg.WEIGHT_DECAY,
            nesterov=True)
    
    def update(self, minibatch):
        image, mask, label, domain = minibatch
        self.optimizer.zero_grad()

        output = self.network(image)
        loss = F.cross_entropy(output, label)
        
        loss.backward()
        self.optimizer.step()
        return {'loss':loss}
    
    def validate(self, val_loader, test_loader, writer):
        val_auc = -1
        test_auc = -1
        if self.epoch <= self.cfg.EPOCHS:
            val_auc, val_loss = algorithm_validate(self, val_loader, writer, self.epoch, 'val')
            test_auc, test_loss = algorithm_validate(self, test_loader, writer, self.epoch, 'test')
            if self.epoch == self.cfg.EPOCHS:
                self.epoch += 1
        else:
            test_auc, test_loss = algorithm_validate(self, test_loader, writer, self.cfg.EPOCHS + self.cfg.VAL_EPOCH, 'test')
            logging.info('Best performance on test domain(s): {}'.format(test_auc))
            
        return val_auc, test_auc
    
    def save_model(self, log_path):
        logging.info("Saving best model...")
        torch.save(self.network.state_dict(), os.path.join(log_path, 'best_model.pth'))
    
    def renew_model(self, log_path):
        net_path = os.path.join(log_path, 'best_model.pth')
        self.network.load_state_dict(torch.load(net_path))
    
    def predict(self, x):
        return self.network(x)
    
class CABNet(ERM):
    
    def __init__(self, num_classes, cfg):
        super(CABNet, self).__init__(num_classes, cfg)
        
class MixStyleNet(ERM):
    
    def __init__(self, num_classes, cfg):
        super(MixStyleNet, self).__init__(num_classes, cfg)
        
class MixupNet(ERM):
    
    def __init__(self, num_classes, cfg):
        super(MixupNet, self).__init__(num_classes, cfg)
        self.criterion_CE = torch.nn.CrossEntropyLoss()
    
    def update(self, minibatch, env_feats=None):
        image, mask, label, domain = minibatch
        self.optimizer.zero_grad()

        inputs, targets_a, targets_b, lam = self.mixup_data(image, label)
        outputs = self.predict(inputs)
        loss = self.mixup_criterion(self.criterion_CE, outputs, targets_a, targets_b, lam)
        
        loss.backward()
        self.optimizer.step()

        return {'loss':loss}
    
    def mixup_data(self, x, y, alpha=1.0, use_cuda=True):
        '''Returns mixed inputs, pairs of targets, and lambda'''
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1

        batch_size = x.size()[0]
        if use_cuda:
            index = torch.randperm(batch_size).cuda()
        else:
            index = torch.randperm(batch_size)

        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        
        return mixed_x, y_a, y_b, lam
    
    def mixup_criterion(self, criterion, pred, y_a, y_b, lam):
        return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
    
class Fishr(ERM):
    def __init__(self, num_classes, cfg):
        super(Fishr, self).__init__(num_classes, cfg)
        
        self.num_groups = cfg.FISHR.NUM_GROUPS

        self.network = models.get_net(cfg)
        self.classifier = extend(
            models.get_classifier(self.network._out_features, cfg)
        )
        self.optimizer = None
        
        self.register_buffer("update_count", torch.tensor([0]))
        self.bce_extended = extend(nn.CrossEntropyLoss(reduction='none'))
        self.ema_per_domain = [
            misc.MovingAverage(cfg.FISHR.EMA, oneminusema_correction=True)
            for _ in range(self.num_groups)
        ]  
        self._init_optimizer()
    
    def _init_optimizer(self):
        self.optimizer = torch.optim.SGD(
            list(self.network.parameters()) + list(self.classifier.parameters()),
            lr = self.cfg.LEARNING_RATE,
            momentum = self.cfg.MOMENTUM,
            weight_decay = self.cfg.WEIGHT_DECAY,
            nesterov=True)
        
    def update(self, minibatch):
        image, mask, label, domain = minibatch
        #self.network.train()

        all_x = image
        all_y = label
        
        len_minibatches = [image.shape[0]]
        
        all_z = self.network(all_x)
        all_logits = self.classifier(all_z)

        penalty = self.compute_fishr_penalty(all_logits, all_y, len_minibatches)
        all_nll = F.cross_entropy(all_logits, all_y)

        penalty_weight = 0
        if self.update_count >= self.cfg.FISHR.PENALTY_ANNEAL_ITERS:
            penalty_weight = self.cfg.FISHR.LAMBDA
            if self.update_count == self.cfg.FISHR.PENALTY_ANNEAL_ITERS != 0:
                # Reset Adam as in IRM or V-REx, because it may not like the sharp jump in
                # gradient magnitudes that happens at this step.
                self._init_optimizer()
        self.update_count += 1

        objective = all_nll + penalty_weight * penalty
        self.optimizer.zero_grad()
        objective.backward()
        self.optimizer.step()

        return {'loss': objective.item(), 'nll': all_nll.item(), 'penalty': penalty.item()}

    def compute_fishr_penalty(self, all_logits, all_y, len_minibatches):
        dict_grads = self._get_grads(all_logits, all_y)
        grads_var_per_domain = self._get_grads_var_per_domain(dict_grads, len_minibatches)
        return self._compute_distance_grads_var(grads_var_per_domain)

    def _get_grads(self, logits, y):
        self.optimizer.zero_grad()
        loss = self.bce_extended(logits, y).sum()
        with backpack(BatchGrad()):
            loss.backward(inputs=list(self.classifier.parameters()), retain_graph=True, create_graph=True)
            #torch.autograd.grad(outputs=loss,inputs=list(self.classifier.parameters()),retain_graph=True, create_graph=True)
            
        # compute individual grads for all samples across all domains simultaneously
        dict_grads = OrderedDict(
            [
                (name, weights.grad_batch.clone().view(weights.grad_batch.size(0), -1))
                for name, weights in self.classifier.named_parameters()
            ]
        )
        return dict_grads

    def _get_grads_var_per_domain(self, dict_grads, len_minibatches):
        # grads var per domain
        grads_var_per_domain = [{} for _ in range(self.num_groups)]
        for name, _grads in dict_grads.items():
            all_idx = 0
            for domain_id, bsize in enumerate(len_minibatches):
                env_grads = _grads[all_idx:all_idx + bsize]
                all_idx += bsize
                env_mean = env_grads.mean(dim=0, keepdim=True)
                env_grads_centered = env_grads - env_mean
                grads_var_per_domain[domain_id][name] = (env_grads_centered).pow(2).mean(dim=0)

        # moving average
        for domain_id in range(self.num_groups):
            grads_var_per_domain[domain_id] = self.ema_per_domain[domain_id].update(
                grads_var_per_domain[domain_id]
            )

        return grads_var_per_domain

    def _compute_distance_grads_var(self, grads_var_per_domain):

        # compute gradient variances averaged across domains
        grads_var = OrderedDict(
            [
                (
                    name,
                    torch.stack(
                        [
                            grads_var_per_domain[domain_id][name]
                            for domain_id in range(self.num_groups)
                        ],
                        dim=0
                    ).mean(dim=0)
                )
                for name in grads_var_per_domain[0].keys()
            ]
        )

        penalty = 0
        for domain_id in range(self.num_groups):
            penalty += self.l2_between_dicts(grads_var_per_domain[domain_id], grads_var)
        return penalty / self.num_groups

    def l2_between_dicts(self, dict_1, dict_2):
        assert len(dict_1) == len(dict_2)
        dict_1_values = [dict_1[key] for key in sorted(dict_1.keys())]
        dict_2_values = [dict_2[key] for key in sorted(dict_1.keys())]
        return (
            torch.cat(tuple([t.view(-1) for t in dict_1_values])) -
            torch.cat(tuple([t.view(-1) for t in dict_2_values]))
        ).pow(2).mean()

# DRGen is built based on Fishr method

class DRGen(Algorithm):
    '''
    Refer to the paper 'DRGen: Domain Generalization in Diabetic Retinopathy Classification' 
    https://link.springer.com/chapter/10.1007/978-3-031-16434-7_61
    
    '''
    def __init__(self, num_classes, cfg):
        super(DRGen, self).__init__(num_classes, cfg)
        algorithm_class = get_algorithm_class('Fishr')
        self.algorithm = algorithm_class(cfg.DATASET.NUM_CLASSES, cfg)
        self.optimizer = self.algorithm.optimizer
        
        self.swad_algorithm = AveragedModel(self.algorithm)
        self.swad_algorithm.cuda()
        #swad_cls = getattr(swad_module, 'LossValley')
        #swad_cls = LossValley()
        self.swad = LossValley(None, cfg.DRGEN.N_CONVERGENCE, cfg.DRGEN.N_TOLERANCE, cfg.DRGEN.TOLERANCE_RATIO)
        
    def update(self, minibatch):
        loss_dict_iter = self.algorithm.update(minibatch)
        if self.swad:
            self.swad_algorithm.update_parameters(self.algorithm, step = self.epoch)
        return loss_dict_iter
    
    def validate(self, val_loader, test_loader, writer):
        swad_val_auc = -1
        swad_auc = -1
        if self.epoch <= self.cfg.EPOCHS:
            val_auc, val_loss = algorithm_validate(self.algorithm, val_loader, writer, self.epoch, 'val(Fishr)')
            test_auc, test_loss = algorithm_validate(self.algorithm, test_loader, writer, self.epoch, 'test(Fishr)')

            if self.swad:
                def prt_results_fn(results):
                    print(results)

                self.swad.update_and_evaluate(
                    self.swad_algorithm, val_auc, val_loss, prt_results_fn
                )
                
                if self.epoch != self.cfg.EPOCHS:
                    self.swad_algorithm = self.swad.get_final_model()
                    swad_val_auc, swad_val_loss = algorithm_validate(self.swad_algorithm, val_loader, writer, self.epoch, 'val')
                    swad_auc, swad_loss = algorithm_validate(self.swad_algorithm, test_loader, writer, self.epoch, 'test')
                    
                    if hasattr(self.swad, "dead_valley") and self.swad.dead_valley:
                        logging.info("SWAD valley is dead -> not stop !")
                        #break
                    
                    self.swad_algorithm = AveragedModel(self.algorithm)  # reset
            
            if self.epoch == self.cfg.EPOCHS:
                self.epoch += 1
                
        else:
            self.swad_algorithm = self.swad.get_final_model()
            logging.warning("Evaluate SWAD ...")
            swad_auc, swad_loss = algorithm_validate(self.swad_algorithm, test_loader, writer, self.cfg.EPOCHS + self.cfg.VAL_EPOCH , 'test')
            logging.info('(last) swad test auc: {}  loss: {}'.format(swad_auc,swad_loss))
            
        return swad_val_auc, swad_auc    
        
    def save_model(self, log_path):
        self.algorithm.save_model(log_path)
    
    def renew_model(self, log_path):
        self.algorithm.renew_model(log_path)
    
    def predict(self, x):
        return self.swad_algorithm.predict(x)