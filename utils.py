
import torch
import torch_dct as dct

#See JPEG standard for Q matrix scaling
def op(value, qf):
    if qf < 50:
        return (50 + ((5000.0/qf) * value))/100.0
    else:
        return (50 + ((200.0 - 2*qf) * value)) / 100.0

#See JPEG standard for Q matrix scaling
def q_table(lc, qf):
    w,h = lc.shape[0], lc.shape[1]
    for i in range(w):
        for j in range(h):
            lc[i][j] = op(lc[i][j], qf)
    return lc

#See JPEG standard for luminance Q matrix
def loch_matrix():
    return torch.tensor([[16, 11, 10, 16, 24, 40, 51, 61],
                      [12, 12, 14, 19, 26, 58, 60, 55],
                      [14, 13, 16, 24, 40, 57, 69, 56],
                      [14,17,22,29,51,87,80,62],
                      [18,22,37,56,68,109,103,77],
                      [24,35,55,64,81,104,113,92],
                      [49,64,78,87,103,121,120,101],
                      [72,92,95,98,112,100,103,99]
                     ], dtype=torch.float).cuda()

"""
#OLD SLOWER WAY

def count_dct_zeros(x):
    lc = loch_matrix()
    lc = torch.floor(q_table(lc, 10))
    aux = x*255.0 - 128.0
    count = 0
    for i in range(int((x.shape[2]) / 8)):
        for j in range(int((x.shape[3]) / 8)):
            a = dct.dct_2d(aux[:, :, i * 8:i * 8 + 8, j * 8:j * 8 + 8], norm='ortho')
            b = torch.abs(a) - lc
            b = torch.relu(b)
            count += (b/(b+0.00001)).sum(dim=(2,3))
    return torch.mean(count) #batch mean
"""

#Takes x (an image) as input, looks at each 8x8 block individually, estimates the non-zero DCT coeffs, sums for each blocks
#then summs all blocks count
#This is faster using  torch.fold

def count_dct_zeros(x, qf):
    #Define lochelechser luminance matrix (see JPEG standard)
    lc = loch_matrix()
    lc = torch.floor(q_table(lc, qf))
    #Shift pixels values (as in JPEG)
    aux = x*255.0 - 128.0
    #Stacks all image 8x8 blocks
    a = torch.nn.functional.unfold(aux, 8, stride=8)
    #Swaps spatial dimensions axis in correct order to perform dct
    b = torch.transpose(a, 1, 2)
    b = b.view([a.shape[0], a.shape[-1], 8, 8])
    #Here b dimensions are [BATCH_SIZE, IMAGE AMOUNT OF 8X8BLOCKS, HEIGHT (8), WIDTH (8))
    #Now do DCT in the last 2 dimensions of b
    b = dct.dct_2d(b, norm='ortho')
    #See Thesis for estimation rate method
    b = torch.abs(b)*2 - lc
    b = torch.relu(b)
    count = (b/(b+0.00001)).sum(dim=(1,2,3))
    return torch.mean(count) #batch mean