
import torchvision.transforms as transforms
from metrics import get_metrics
import dataset
import PIL.Image as Image
from models.nets import *
import subprocess
import xlwt
import os

#Assumes full directory is ../dataset/DATASET
DATASET = 'MPEG'

#FOR INDEPENDET INFFERENCE
#in PATH have to be the trained CR and SR state_dicts
PATH = './checkpoint/L1e-8_QF10.pth'

#Codes an image with JPEG,  computes its bpp and returns coded image in tensor format
def encode(img, qf):
    size = img.shape
    a = transforms.ToPILImage(mode='L')(img[0][0].cpu())
    a.save('img/ds.pgm')
    cmd = ['jpeg', '-q', str(qf), 'img/ds.pgm', 'img/ds_qf55.jpg']  # command to run
    output = subprocess.call(cmd, stdout=subprocess.DEVNULL)  # run the command
    b = Image.open('img/ds_qf55.jpg')
    bpp = os.path.getsize('img/ds_qf55.jpg')*8/4/b.width/b.height
    out = torch.zeros((size))
    out[0][0] = transforms.ToTensor()(b)
    return out.cuda(), bpp


#Goes to path, loads both models and infers for full-resolution image, updates excel.
def eval(path, sheet, epoch):
    dset = dataset.get_dataset(1, 'MPEG', False)
    cr = CNNCRluma().cuda()
    sr = CNNSRluma().cuda()
    print(torch.load(path)['epoch'])
    #loads net weigths
    cr.load_state_dict(torch.load(path)['cr'])
    sr.load_state_dict(torch.load(path)['sr'])
    cr.eval()
    sr.eval()
    total, rate = 0.0, 0.0
    for iteration, data in enumerate(dset, 1):
        input, name = data[0].cuda(), data[1]
        with torch.no_grad():
            #FORWARD PASS========
            #down-sample input
            ds, _ = cr(input)
            ds = ds.clamp(0,1) #Because of intperolation
            #code down-sampled input
            coded, bpp = encode(ds, 25)
            #up-sampled decoded image
            us = sr(coded).clamp(0,1)
            #====================
            out = transforms.ToPILImage(mode='L')(us[0][0].cpu())
            gt = transforms.ToPILImage(mode='L')(input[0][0].cpu())

            psnr = get_metrics(gt, out, False)[0]
            total += psnr
            rate += bpp
            print('Bpp: {} --- PSNR: {}'.format(bpp, psnr))
            torch.cuda.empty_cache()
    print(rate/len(dset))
    print(total/len(dset))
    #sheet.write(epoch+1, 7, rate/len(dset))
    #sheet.write(epoch+1, 8, total/len(dset))

#FOR INDEPENDET INFFERENCE
#in path have to be the trained CR and SR state_dicts
if __name__ == '__main__':
    path = PATH
	eval(path, '', 0)