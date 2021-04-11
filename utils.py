import numpy as np

def maskToRle(mask):
    rows, cols = mask.shape
    counts = []
    cc = 0
    
    dummy = mask[0][0]        
        
    if dummy == 1:
        counts.append(0)
        
    for i in range(0,cols):
        for j in range(0,rows):
            if mask[j][i] == dummy:
                cc = cc + 1
            else:
                dummy = mask[j][i]
                counts.append(cc)
                cc = 1
                
    counts.append(cc)     
    
    size = [rows,cols]
        
    annotation = {
            'counts': counts,
            'size': size
    }    
    
    return annotation


def RleToMask(rle):
    rows = rle['size'][0]
    cols = rle['size'][1]
    string = rle['counts']
    
    mask = np.zeros(rows*cols)
    dummy = 0
    j = 0
    
    for indx in range(0,len(string)):
        value = string[indx]
        
        if value == 0:
            dummy = 1
            continue
        
        while value > 0:
            mask[j] = dummy
            j += 1
            value -= 1
        
        if dummy == 0:
            dummy = 1
        else:
            dummy = 0
            
    res_mask = np.zeros((rows,cols))
    x = 0
    
    for i in range(0,cols):
        for j in range(0,rows):
            res_mask[j][i] = mask[x]
            x = x+1
    
    return res_mask.astype(int)