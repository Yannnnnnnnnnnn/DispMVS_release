
def eval(gt,mask,pr):

    diff = (gt - pr).abs()
    diff = diff[mask>0]

    mask = mask.sum()

    acc_1mm = 100*( diff<1 ).sum() 
    acc_2mm = 100*( diff<2 ).sum()
    acc_4mm = 100*( diff<4 ).sum() 

    return diff.sum(),acc_1mm,acc_2mm,acc_4mm,mask
