import torch
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
import logging
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import normalize
import seaborn as sns

# validate the algorithm by AUC, accuracy and f1 score on val/test datasets

def algorithm_validate(algorithm, data_loader, writer, epoch, val_type):
    algorithm.eval()
    criterion = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        softmax = torch.nn.Softmax(dim=1)
        loss = 0
        label_list = []
        output_list = []
        pred_list = []

        for image, label, domain, _ in data_loader:
            image = image.cuda()
            label = label.cuda().long()

            output = algorithm.predict(image)
            loss += criterion(output, label).item()

            _, pred = torch.max(output, 1)
            output_sf = softmax(output)

            label_list.append(label.cpu().data.numpy())
            pred_list.append(pred.cpu().data.numpy())
            output_list.append(output_sf.cpu().data.numpy())
        
        label = [item for sublist in label_list for item in sublist]
        pred = [item for sublist in pred_list for item in sublist]
        output = [item for sublist in output_list for item in sublist]

        acc = accuracy_score(label, pred)
        f1 = f1_score(label, pred, average='macro')
    
        auc_ovo = roc_auc_score(label, output, average='macro', multi_class='ovo')

        loss = loss / len(data_loader)

        if val_type in ['val', 'test']:
            writer.add_scalar('info/{}_accuracy'.format(val_type), acc, epoch)
            writer.add_scalar('info/{}_loss'.format(val_type), loss, epoch)
            writer.add_scalar('info/{}_auc_ovo'.format(val_type), auc_ovo, epoch)
            writer.add_scalar('info/{}_f1'.format(val_type), f1, epoch)     
                
            logging.info('{} - epoch: {}, loss: {}, acc: {}, auc: {}, F1: {}.'.format
            (val_type, epoch, loss, acc, auc_ovo, f1))

    algorithm.train()
    # return auc_ovo, loss
    return acc, loss

def algorithm_validate_iw(algorithm, data_loader, writer, epoch, val_type):
    algorithm.eval()
    criterion = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        softmax = torch.nn.Softmax(dim=1)
        loss = 0
        label_list = []
        output_list = []
        pred_list = []

        for image, label, domain, _ in data_loader:
            image = image.cuda()
            label = label.cuda().long()

            # output, iw_loss, sw_loss = algorithm.predict(image)
            # loss += criterion(output, label).item()
            # loss += iw_loss * 1e-6
            # loss += sw_loss
            output, iw_loss, _ = algorithm.predict(image)
            loss += iw_loss * 1e-6

            _, pred = torch.max(output, 1)
            output_sf = softmax(output)

            label_list.append(label.cpu().data.numpy())
            pred_list.append(pred.cpu().data.numpy())
            output_list.append(output_sf.cpu().data.numpy())
        
        label = [item for sublist in label_list for item in sublist]
        pred = [item for sublist in pred_list for item in sublist]
        output = [item for sublist in output_list for item in sublist]

        acc = accuracy_score(label, pred)
        f1 = f1_score(label, pred, average='macro')
    
        auc_ovo = roc_auc_score(label, output, average='macro', multi_class='ovo')

        loss = loss / len(data_loader)

        if val_type in ['val', 'test']:
            writer.add_scalar('info/{}_accuracy'.format(val_type), acc, epoch)
            writer.add_scalar('info/{}_loss'.format(val_type), loss, epoch)
            writer.add_scalar('info/{}_auc_ovo'.format(val_type), auc_ovo, epoch)
            writer.add_scalar('info/{}_f1'.format(val_type), f1, epoch)     
                
            logging.info('{} - epoch: {}, loss: {}, acc: {}, auc: {}, F1: {}.'.format
            (val_type, epoch, loss, acc, auc_ovo, f1))

    algorithm.train()
    return acc, loss

def algorithm_eval_iw(algorithm, data_loader, writer, epoch, val_type):
    # algorithm.eval()
    criterion = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        softmax = torch.nn.Softmax(dim=1)
        loss = 0
        label_list = []
        output_list = []
        pred_list = []

        for image, label, domain, _ in tqdm(data_loader):
            image = image.cuda()
            label = label.cuda().long()

            output, iw_loss, _ = algorithm.predict(image)
            # loss += criterion(output, label).item()
            # loss += iw_loss * 1e-6
            # loss += sw_loss

            _, pred = torch.max(output, 1)
            output_sf = softmax(output)

            label_list.append(label.cpu().data.numpy())
            pred_list.append(pred.cpu().data.numpy())
            output_list.append(output_sf.cpu().data.numpy())
        
        label = [item for sublist in label_list for item in sublist]
        pred = [item for sublist in pred_list for item in sublist]
        output = [item for sublist in output_list for item in sublist]

        acc = accuracy_score(label, pred)
        f1 = f1_score(label, pred, average='macro')
        auc_ovo = roc_auc_score(label, output, average='macro', multi_class='ovo')
        num_ = np.zeros(5)
        correct_ = np.zeros(5)
        for i, l in enumerate(label):
            num_[int(l)] += 1
            if int(pred[i]) == int(l):
                correct_[int(l)] += 1
        accs = correct_ / num_

        # label_ = np.array(label)
        # pred_ = np.array(pred)
        # cm = confusion_matrix(label_, pred_)
        # cm = normalize(cm, axis=1, norm='l1')
        # # cm = (cm - cm.min()) / (cm.max() - cm.min())
        # plt.figure(figsize=(10, 6))
        # sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", xticklabels=['nodr', 'mild_npdr', 'moderate_npdr', 'severe_npdr', 'pdr'],
        #             yticklabels=['nodr', 'mild_npdr', 'moderate_npdr', 'severe_npdr', 'pdr'])
        # plt.xlabel("Pred")
        # plt.ylabel("Label")
        # plt.title("ESDG-APTOS")
        # # plt.show()
        # plt.savefig('vis/cm_ESDG-APTOS.png')
        # exit()

        # loss = loss / len(data_loader)

        if val_type in ['val', 'test']:
            writer.add_scalar('info/{}_accuracy'.format(val_type), acc, epoch)
            # writer.add_scalar('info/{}_loss'.format(val_type), loss, epoch)
            writer.add_scalar('info/{}_auc_ovo'.format(val_type), auc_ovo, epoch)
            writer.add_scalar('info/{}_f1'.format(val_type), f1, epoch)             
            logging.info('{} - epoch: {}, loss: {}, acc: {}, auc: {}, F1: {}.\n accs: {}'.format
            (val_type, epoch, loss, acc, auc_ovo, f1, accs))

    return acc


def algorithm_eval_mu(algorithm, data_loader, epoch, val_type):
    # algorithm.eval()
    with torch.no_grad():
        i = 0
        for image, label, domain, _ in tqdm(data_loader):
            image = image.cuda()
            x = image
            img_mean = x.mean(dim=[2,3], keepdim=True) # B,C,1,1 mu
            img_std = x.std(dim=[2,3], keepdim=True) + 1e-7# B,C,1,1 sigma
            if i == 0:
                mu = img_mean.cpu().numpy()
                sigma = img_std.cpu().numpy()
            else:
                mu = np.concatenate((mu, img_mean.detach().cpu().numpy()), axis=0)
                sigma = np.concatenate((sigma, img_std.detach().cpu().numpy()), axis=0)
            i += 1

    return mu, sigma


def algorithm_eval_tsne(algorithm, data_loader, epoch, val_type):
    # algorithm.eval()
    with torch.no_grad():
        i = 0
        for image, label, domain, _ in tqdm(data_loader):
            image = image.cuda()
            x = image
            img_mean = x.mean(dim=[2,3], keepdim=True) # B,C,1,1 mu
            img_std = x.std(dim=[2,3], keepdim=True) + 1e-7# B,C,1,1 sigma
            label = label.cuda().long()

            output, iw_loss, _ = algorithm.predict(image)

            if i == 0:
                mu = img_mean.cpu().numpy()
                sigma = img_std.cpu().numpy()
                feats = output.cpu().numpy()
                labels = label.cpu().numpy()
            else:
                mu = np.concatenate((mu, img_mean.detach().cpu().numpy()), axis=0)
                sigma = np.concatenate((sigma, img_std.detach().cpu().numpy()), axis=0)
                feats = np.concatenate((feats, output.detach().cpu().numpy()), axis=0)
                labels = np.concatenate((labels, label.detach().cpu().numpy()), axis=0)
            i += 1

    return feats, labels


def algorithm_eval_heat(algorithm, data_loader, epoch, val_type):
    # algorithm.eval()
    with torch.no_grad():
        i = 0
        for image, label, domain, _ in tqdm(data_loader):
            image = image.cuda()
            x = image
            label = label.cuda().long()

            output, iw_loss, feat_, feat = algorithm.predict(image)
            B, N, C = feat.shape[:]
            feat = feat.view(B, int(np.sqrt(N)), -1, C)
            feat = feat.mean(dim=-1)

            if i == 0:
                feats = feat.cpu().numpy()
            else:
                feats = np.concatenate((feats, feat.detach().cpu().numpy()), axis=0)
            i += 1

    return feats