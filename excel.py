

#Writes to excel sheet
def update_excel_sheet(sheet, row, column, name, metric, bpp, QP):
    psnr, ssim, niqe = metric
    print(name, ': ', psnr, '-',ssim, '-', niqe, '-', bpp, '-', QP)
    sheet.write(row, column+1, name)
    sheet.write(row, column+2, psnr)
    sheet.write(row, column+3, ssim)
    sheet.write(row, column+4, float(niqe))
    sheet.write(row, column+5, bpp)
    sheet.write(row, column+6, QP)

#Initialize excel sheet with some labels
def init_excel(sheet):
    sheet.write(0, 1, 'name')
    sheet.write(0, 2, 'psnr')
    sheet.write(0, 3, 'ssim')
    sheet.write(0, 4, 'niqe')
    sheet.write(0, 5, 'bpp')
    sheet.write(0, 6, 'QP')
    return sheet
