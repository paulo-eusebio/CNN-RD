
from utils import *
import torch.optim as optim
from models.nets import *
from eval import eval
import time, os, xlwt
from dataset import get_dataset


#HYPER-PARAMS
NUM_EPOCHS = 30
BATCH_SIZE = 16
LR = 0.0001
#INPUT AND TARGET FOLDERS ARE THE SAME
#This datasets contained 120k sample images with 96x96 size
#Assumes full directory is ../dataset/DATASET
DATASET = 'DIV2Kcrop96'

#Settings
PENALTY = 1e-8 #for the loss function
QF = 1 #for counting DCT zeros

"""
    Main: Loads trainig dataset, declares CNNs and GPUs. Starts training

"""

def main(num_epochs, dataset, lr, batch_size, sheet):
    elapsed_time = 0.0
    print("===> Loading datasets")
    training_data_loader = get_dataset(batch_size, dataset, shuffle=True)
    print("===> Building model")
    print("===> Setting GPU")
    #Two nets. One for before encoding, other for after encoding.
    SR = CNNSRluma().cuda()
    CR = CNNCRluma().cuda()
    criterion = nn.MSELoss().cuda()
    print("===> Setting Optimizer")
    #Both net parameters are going to be trained, thus send them to the optimizer
    optimizer = optim.Adam(list(CR.parameters())+list(SR.parameters()), lr=lr, betas=(0.9,0.9), eps=(10**-8))
    print("===> Training")
    #train for num_epochs
    for epoch in range(num_epochs):
        start_time = time.time()
        train(training_data_loader, optimizer, CR, SR, criterion, epoch, sheet)
        path = save_checkpoint(CR,SR, optimizer, epoch)
        #Eval trained nets every epoch
        eval(path, sheet, epoch)
        elapsed_time += time.time() - start_time
        print("Accumulated training time (mins) = {:f}".format(elapsed_time/60.0))

"""
    Train: Has the actual training process. This method is called every training epoch
           Forward pass+backward pass+updates+epoch log
           
"""

def train(training_data_loader, optimizer, cr, sr, criterion, epoch, sheet):
    #adjusting learning rate
    if (epoch+1) % 10 == 0:
        for param_group in optimizer.param_groups:
            param_group['lr'] /= 2

    print("Epoch = {}, lr = {}".format(epoch, optimizer.param_groups[0]['lr']))

    cr.train()
    sr.train()
    #===Vars for logs
    running_loss = 0.0
    bpp_avg = 0.0
    mse_avg = 0.0
    #====
    penalty = PENALTY
    qf = QF
    for iteration, batch in enumerate(training_data_loader, 1):
        #Clear history
        optimizer.zero_grad()
        #Input and target are the same
        input, target = batch[0].cuda(), batch[1].cuda()
        #===============Forward Pass
        #Down-sampling
        lr,_ = cr(input) #
        #Up-sampling
        out = sr(lr)
        #================Loss
        #Distortion loss
        mse_loss = criterion(out, target)
        #Estimate JPEG rate based on DCT zeros
        non_zeros = count_dct_zeros(lr, qf=qf)
        #Compute global loss
        loss = mse_loss + non_zeros*penalty
        #================Backward Pass
        loss.backward()
        #================Update Params
        optimizer.step()
        #================LOGS
        running_loss += loss.item()
        bpp_avg += non_zeros.item()
        mse_avg += mse_loss.item()
        if iteration%(len(training_data_loader)) == 0:
            print("===> Epoch[{}]({}/{}): Net Loss: {:.10f}".format(epoch, iteration, len(training_data_loader), mse_avg/len(training_data_loader)))
            print("===> Epoch[{}]({}/{}): Net Loss: {:.10f}".format(epoch, iteration, len(training_data_loader), bpp_avg/len(training_data_loader)))
            print("===> Epoch[{}]({}/{}): Net Loss: {:.10f}".format(epoch, iteration, len(training_data_loader), running_loss/len(training_data_loader)))
            sheet.write(epoch+1, 1, epoch+1)
            sheet.write(epoch+1, 2, bpp_avg/len(training_data_loader))
            sheet.write(epoch+1, 3, bpp_avg*penalty/len(training_data_loader))
            sheet.write(epoch+1, 4, mse_avg/len(training_data_loader))
            sheet.write(epoch+1, 5, running_loss/len(training_data_loader))
            running_loss = 0.0
            bpp_avg = 0.0


#Save models state dictionary only, i.e. only param values
def save_checkpoint(cr,sr, optim, epoch):
    path = 'checkpoint/L{}_QF{}.pth'.format(PENALTY, QF)
    if not os.path.exists("checkpoint/"):
        os.makedirs("checkpoint/")
    torch.save({
                'epoch': epoch,
                'cr': cr.state_dict(),
                'sr': sr.state_dict(),
                'optim': optim.state_dict()
                }, path)
    print("Checkpoint saved to {}".format(path))
    return path



if __name__ == "__main__":
    #=======Set up excel
    wb = xlwt.Workbook()
    sheet = wb.add_sheet('Sheet1')
    sheet.write(0, 1, 'Epoch')
    sheet.write(0, 2, 'Average MSE bicubic loss)')
   # sheet.write(0, 3, 'Average Rate*Penalty')
    sheet.write(0, 4, 'Average MSE Loss')
    sheet.write(0, 7, 'Average Rate')
    sheet.write(0, 8, 'Average PSNR')
    #=====================
    main(NUM_EPOCHS, DATASET, LR, BATCH_SIZE, sheet)
    wb.save('PEN:{}_QF{}.xls'.format(PENALTY,QF))
