from __future__ import print_function
import argparse
import os
import numpy as np
from PIL import Image
import torch
from torch.utils import data
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch import nn
import torchvision.utils as vutils
from torch.autograd import Variable
import torch.nn.functional as F
from net import NetC,NetS
from LoadData import Dataset, loader, Dataset_val
from logger import Logger
import matplotlib.pyplot as plt
from skimage import filters,color


import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def get_arguments():
    parser = argparse.ArgumentParser(description='Example')
    parser.add_argument('--batchSize', type=int, default=10, help='training batch size')
    parser.add_argument('--niter', type=int, default=500, help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.0002, help='Learning Rate. Default=0.02')
    parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use, for now it only supports one GPU')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--decay', type=float, default=0.5, help='Learning rate decay. default=0.5')
    parser.add_argument('--cuda', type=bool,default=True, help='using GPU or not')
    parser.add_argument('--seed', type=int, default=666, help='random seed to use. Default=1111')
    parser.add_argument('--outpath', default='./outputs/Dense_Residual_block', help='folder to output images and model checkpoints')
    parser.add_argument('--ckpt', default='./ckpt/Dense_Residual_block',
                        help='folder to output checkpoints')
    parser.add_argument('--a', type=float, default=0.5, help='Lepe loss index')
    parser.add_argument('--b', type=float, default=0.5, help=' LJac loss index')
    return parser.parse_args()

opt=get_arguments()
print(opt)
try:
    os.makedirs(opt.outpath)
except OSError:
    pass
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def dice_loss(input,target):
    num=input*target
    num=torch.sum(num,dim=2)
    num=torch.sum(num,dim=2)

    den1=input*input
    den1=torch.sum(den1,dim=2)
    den1=torch.sum(den1,dim=2)

    den2=target*target
    den2=torch.sum(den2,dim=2)
    den2=torch.sum(den2,dim=2)

    Dsc=2*(num/(den1+den2))

    dice_total=1-1*torch.sum(Dsc)/Dsc.size(0)
    return dice_total

def EPE(predicted_edge, gt_edge, sparse=False, mean=True):
    EPE_map = torch.norm(gt_edge-predicted_edge,2,1)
    if sparse:
        EPE_map = EPE_map[gt_edge != 0]
    if mean:
        return EPE_map.mean()
    else:
        return EPE_map.sum()
def getEdge(batch):
    edgeslist=[]
    for kk in range(batch.size(0)):
        x=batch[kk]
        x=x.cpu().data.numpy()
        x=np.array(x,dtype=np.uint8)
        if len(x.shape)>2:
            x=np.transpose(x,(1,2,0))
            x=color.rgb2gray(x)
        edges = filters.sobel(x)
        edgeslist.append(edges)
    edgeslist=np.array(edgeslist)
    edgeslist=torch.Tensor(edgeslist).cuda()
    edgeslist=Variable(edgeslist)
    return  edgeslist

def main():
    from Dense-residual-block import NetC,NetS
    cuda = opt.cuda
    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")
    torch.manual_seed(opt.seed)
    if cuda:
        torch.cuda.manual_seed(opt.seed)
    cudnn.benchmark = True
    n_layers_list = [4, 5, 7, 10, 12, 15, 12, 10, 7, 5, 4]
    NetS = NetS(n_layers_list, 5)
    print(NetS)
    NetC = NetC(ngpu = opt.ngpu)
    print(NetC)
    if cuda:
        NetS = NetS.cuda()
        NetC = NetC.cuda()
    lr = opt.lr
    decay = opt.decay
    optimizerG = optim.Adam(NetS.parameters(), lr=lr, betas=(opt.beta1, 0.999))
    optimizerD = optim.Adam(NetC.parameters(), lr=lr, betas=(opt.beta1, 0.999))
    dataloader = loader(Dataset('./'),opt.batchSize)
    print(len(dataloader))
    dataloader_val = loader(Dataset_val('./'), opt.batchSize)
    print(len(dataloader_val))
    logger=Logger('./logs/Dense_Residual_block')
    max_Jac = 0
    NetS.train()
    history = {split: {'epoch': [], 'Loss_D': [], 'Loss_G_joint': []}
               for split in ('train', 'val')}

    history1 = {split: {'epoch': [], 'Jac': [], 'Dsc': [], 'acc': [],'se':[], 'sp':[]}
                for split in ('train', 'val')}
    for epoch in range(opt.niter):
        for i, data in enumerate(dataloader, 1):
            NetC.zero_grad()
            input, label = Variable(data[0]), Variable(data[1])
            if cuda:
                input = input.cuda()
                target = label.cuda()
            else:
                input=input
                target=label
            target = target.type(torch.FloatTensor)
            target = target.cuda()
            output = NetS(input)
            output = F.sigmoid(output)
            output = output.detach()
            output_masked = input.clone()
            input_mask = input.clone()
            for d in range(3):
                if d==0:
                    output_masked[:, d:, :, :] = input_mask[:, d, :, :].unsqueeze(1) * output
                else:
                    output_masked[:,:d:, :, :] = input_mask[:,d,:,:].unsqueeze(1) * output
            if cuda:
                output_masked = output_masked.cuda()
            result = NetC(output_masked)
            target_masked = input.clone()
            for d in range(3):
                if d==0:
                    target_masked[:,d:,:,:] = input_mask[:,d,:,:].unsqueeze(1) * target
                else:
                    target_masked[:,:d:,:,:] = input_mask[:,d,:,:].unsqueeze(1) * target
            if cuda:
                target_masked = target_masked.cuda()
            target_D = NetC(target_masked)
            target_D.detach()
            loss_D = -torch.mean(torch.abs(result - target_D))
            loss_D.backward()
            optimizerD.step()
            for p in NetC.parameters():
                p.data.clamp_(-0.05, 0.05)
            NetS.zero_grad()
            output = NetS(input)
            output = F.sigmoid(output)
            for d in range(3):
                if d==0:
                    output_masked[:, d:, :, :] = input_mask[:, d, :, :].unsqueeze(1) * output
                else:
                    output_masked[:, :d:, :, :] = input_mask[:, d, :, :].unsqueeze(1) * output
            if cuda:
                output_masked = output_masked.cuda()
            result = NetC(output_masked)
            for d in range(3):
                if d==0:
                    target_masked[:, d:, :, :] = input_mask[:, d, :, :].unsqueeze(1) * target
                else:
                    target_masked[:, :d:, :, :] = input_mask[:, d, :, :].unsqueeze(1) * target
            if cuda:
                target_masked = target_masked.cuda()
            target_G = NetC(target_masked)
            loss_dice = dice_loss(output,target)
            loss_G = torch.mean(torch.abs(result - target_G))
            label_edge = getEdge(target_masked)
            _, pred3 = (torch.max(output_masked, 1))
            pred_edge = getEdge(pred3)
            loss_G_joint = torch.mean(torch.abs(result - target_G)) + opt.a * EPE(label_edge,pred_edge)+opt.b*loss_dice
            loss_G_joint.backward()
            optimizerG.step()
            step = len(dataloader) * epoch + i
            info = {'D_loss': loss_D.data[0], 'G_loss': loss_G.data[0], 'loss_dice': loss_dice.data[0]}
            for tag, value in info.items():
                logger.scalar_summary(tag, value, step)
        print("===> Epoch[{}]({}/{}): Batch Dice: {:.4f}".format(epoch, i, len(dataloader), 1 - loss_dice.data[0]))
        print("===> Epoch[{}]({}/{}): G_Loss: {:.4f}".format(epoch, i, len(dataloader), loss_G.data[0]))
        print("===> Epoch[{}]({}/{}): D_Loss: {:.4f}".format(epoch, i, len(dataloader), loss_D.data[0]))
        vutils.save_image(data[0],
                '%s/input.png' % opt.outpath,
                normalize=True)
        vutils.save_image(data[1],
                '%s/label.png' % opt.outpath,
                normalize=True)
        vutils.save_image(output.data,
                '%s/result.png' % opt.outpath,
                normalize=True)
        if epoch % 10 == 0:
            NetS.eval()

            Jacs, dices, accs,SEs,SPs= [], [],[],[],[]
            for i, data in enumerate(dataloader_val, 1):
                input, gt = Variable(data[0]), Variable(data[1])
                if cuda:
                    input = input.cuda()
                    gt = gt.cuda()
                pred = NetS(input)
                pred[pred < 0.5] = 0
                pred[pred >= 0.5] = 1
                pred = pred.type(torch.LongTensor)
                pred_np = pred.data.cpu().numpy()
                gt = gt.data.cpu().numpy()
                for x in range(input.size()[0]):
                    Jac = np.sum(pred_np[x][gt[x]==1]) / float(np.sum(pred_np[x]) + np.sum(gt[x]) - np.sum(pred_np[x][gt[x]==1]))
                    Dsc = np.sum(pred_np[x][gt[x]==1])*2 / float(np.sum(pred_np[x]) + np.sum(gt[x]))
                    acc = float(np.sum(pred_np[x][gt[x] == 1])+np.sum(pred_np[x][gt[x] == 0])) / float(np.sum(pred_np[x]) + np.sum(gt[x])+np.sum(pred_np[x][gt[x] == 1])+np.sum(pred_np[x][gt[x] == 0]))
                    SE=np.sum(pred_np[x][gt[x]==1])/ float(np.sum(pred_np[x][gt[x]==1])+np.sum(pred_np[x]))
                    SP=np.sum(pred_np[x][gt[x] == 0])/float(np.sum(pred_np[x][gt[x] == 0])+np.sum(gt[x]))

                    Jacs.append(Jac)
                    Dscs.append(Dsc)
                    accs.append(acc)
                    SEs.append(SE)
                    SPs.append(SP)


            NetS.train()
            Jacs = np.array(Jacs, dtype=np.float64)
            Dscs = np.array(dices, dtype=np.float64)
            accs = np.array(accs, dtype=np.float64)
            SEs = np.array(SEs, dtype=np.float64)
            SPs = np.array(SPs, dtype=np.float64)

            mJac = np.mean(Jacs, axis=0)
            mDsc = np.mean(Dscs, axis=0)
            macc = np.mean(accs, axis=0)
            mSE= np.mean(SEs, axis=0)
            mSP = np.mean(SPs, axis=0)

            history1['train']['epoch'].append(epoch)
            history1['train']['Jac'].append(mJac)
            history1['train']['Dsc'].append(mDsc)
            history1['train']['acc'].append(macc)
            history1['train']['SE'].append(mSE)
            history1['train']['SP'].append(mSP)
            print('Plotting  evaluation metrics figure...')
            plt.xlabel('Epoch')
            plt.ylabel('Jac_dice')
            fig = plt.figure()
            plt.plot(history1['train']['epoch'], history1['train']['Jac'], color='b', label='Jac')
            plt.plot(history1['train']['epoch'], history1['train']['Dsc'], color='c', label='Dsc')
            plt.plot(history1['train']['epoch'], history1['train']['acc'], color='y', label='acc')
            plt.plot(history1['train']['epoch'], history1['train']['SE'], color='g', label='SE')
            plt.plot(history1['train']['epoch'], history1['train']['SP'], color='r', label='SP')
            plt.legend()
            fig.savefig('{}/ evaluation metrics.png'.format(opt.ckpt), dpi=200)
            plt.close('all')
            info = {'Jac': mJac, 'DSC': mDsc,'acc':macc, 'SE':mSE, 'SP':mSP}
            for tag, value in info.items():
              logger.scalar_summary(tag, value, epoch)
            print('mJac: {:.4f}'.format(mJac))
            print('mDsc: {:.4f}'.format(mDsc))
            print('macc: {:.4f}'.format(macc))
            print('mSE: {:.4f}'.format(mSE))
            print('mSP: {:.4f}'.format(mSP))
            if mJac > max_Jac:
                max_Jac = mJac
                torch.save(NetS.state_dict(), '%s/NetS_epoch_%d.pth' % (opt.outpath, epoch))
            vutils.save_image(data[0],
                    '%s/input_val.png' % opt.outpath,
                    normalize=True)
            vutils.save_image(data[1],
                    '%s/label_val.png' % opt.outpath,
                    normalize=True)
            pred = pred.type(torch.FloatTensor)
            vutils.save_image(pred.data,
                    '%s/result_val.png' % opt.outpath,
                    normalize=True)
        if epoch % 25 == 0:
            lr = lr*decay
            if lr <= 0.000001:
                lr = 0.000001
            print('Learning Rate: {:.6f}'.format(lr))
            print('Max mJac: {:.4f}'.format(max_Jac))
            optimizerG = optim.Adam(NetS.parameters(), lr=lr, betas=(opt.beta1, 0.999))
            optimizerD = optim.Adam(NetC.parameters(), lr=lr, betas=(opt.beta1, 0.999))

if __name__ == '__main__':
    main()
