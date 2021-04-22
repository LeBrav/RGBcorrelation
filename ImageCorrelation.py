import cv2 as cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy import signal
import time
from scipy.signal import convolve2d, fftconvolve


# LLEGIR DATASET###############################################################################################
'''
dataset links:
    https://www.loc.gov/pictures/related/?va=exact&co=prok&sp=13&q=Glass+negatives.&fi=formats&sg=true&op=EQUAL
1# https://www.loc.gov/pictures/collection/prok/item/2018679063/
2# https://www.loc.gov/pictures/collection/prok/item/2018679003/
3# https://www.loc.gov/pictures/collection/prok/item/2018679297/
4# https://www.loc.gov/pictures/collection/prok/item/2018679473/
5# https://www.loc.gov/pictures/collection/prok/item/2018679608/
'''
def fix_borders(channel, edge_detector=''):
    "https://gyazo.com/c015412c84868f8f0bba9f8729029989"
    if edge_detector == 'sobel':
        #sobel
        img_sobelx = cv2.Sobel(channel,cv2.CV_8U,1,0,ksize=5)
        img_sobely = cv2.Sobel(channel,cv2.CV_8U,0,1,ksize=5)
        edges = img_sobelx + img_sobely
    else:
        #canny
        edges = cv2.Canny(channel,100,200)
        
    edges = edges>0 #convert to binary
    
    suma_columnes = np.sum(edges, axis = 1)
    suma_files = np.sum(edges, axis = 0)
    
    #Per trobar el borde passarem la imatge del canal a una Edge IMg utilitzant Canny, 
    #on obtindrem els contorns de la imatge. Ara volem obtenir els index (np.where), 
    #de la part superiors de la imatge, int(r.shape[1]/10), els quals tinguin mes del 50% dels pixels blancs
    #si una linea te més del 25% de pixels blancs, es un borde
    
    #BORDE ADALT
    index_fila_amunt = np.where(suma_columnes[:int(channel.shape[0]/10)] > int(channel.shape[0]/3))[0]
    if index_fila_amunt.size != 0:
        index_fila_amunt = index_fila_amunt[-1]
        channel = np.delete(channel, range(index_fila_amunt), axis = 0)
    
    #BORDE abaix
    index_fila_avall = np.where(np.flip(suma_columnes[-int(channel.shape[0]/10):]) > int(channel.shape[0]/3))[0]
    if index_fila_avall.size != 0:
        index_fila_avall = index_fila_avall[-1]
        channel = np.delete(channel, range(channel.shape[0]-index_fila_avall, channel.shape[0]), axis = 0)
    
    #BORDE esquerra
    index_col_esquerra = np.where(suma_files[:int(channel.shape[1]/10)] > int(channel.shape[1]/3))[0]
    if index_col_esquerra.size != 0:
        index_col_esquerra = index_col_esquerra[-1]
        channel = np.delete(channel, range(index_col_esquerra), axis = 1)
    
    #BORDE dreta
    index_col_dreta = np.where(suma_files[:int(channel.shape[1]/10)] > int(channel.shape[1]/3))[0]
    if index_col_dreta.size != 0:
        index_col_dreta = index_col_dreta[-1]
        channel = np.delete(channel, range(channel.shape[1]-index_col_dreta, channel.shape[1]), axis = 1)
    

    return channel

def obtainImage(filename):
    img = cv2.imread('dataset/' + filename, 0)
    #PARTIR EN 3 SUBIMATGES###############################################################################################
    size = int(img.shape[0]/3)
    r = img[0:size,:]
    g = img[size:size*2,:]
    b = img[size*2:size*3,:]
    
    plt.subplot(131), plt.imshow(r, "gray"), plt.title("Vermell"), plt.xticks([]), plt.yticks([])
    plt.subplot(132), plt.imshow(g, "gray"), plt.title("Verd"), plt.xticks([]), plt.yticks([])
    plt.subplot(133), plt.imshow(b, "gray"), plt.title("Blau"), plt.xticks([]), plt.yticks([])
    plt.show()
    
    plt.imshow(cv2.cvtColor(cv2.Canny(r,100,200), cv2.COLOR_BGR2RGB)), plt.title("edge detector abans del retall")
    plt.show()
    
    #rgb retallat:
    r = fix_borders(r, 'sobel')
    g = fix_borders(g, 'sobel')
    b = fix_borders(b, 'sobel')
    
    #fem tots els canals de la mateixa mida agafant el mes petit i retallantlos tots
    minim_col = min([r.shape[0], g.shape[0], b.shape[0]])
    minim_row = min([r.shape[1], g.shape[1], b.shape[1]])
    r = r[:minim_col, :minim_row]
    g = g[:minim_col, :minim_row]
    b = b[:minim_col, :minim_row]
    
    plt.imshow(cv2.cvtColor(cv2.Canny(r,100,200), cv2.COLOR_BGR2RGB)), plt.title("edge detector després del retall")
    plt.show()
    
    plt.subplot(131), plt.imshow(r, "gray"), plt.title("Vermell"), plt.xticks([]), plt.yticks([])
    plt.subplot(132), plt.imshow(g, "gray"), plt.title("Verd"), plt.xticks([]), plt.yticks([])
    plt.subplot(133), plt.imshow(b, "gray"), plt.title("Blau"), plt.xticks([]), plt.yticks([])
    plt.show()




    # Crear template per a fer template matching############################################################################
    y, x = b.shape
    template_perc = 0.90
    retallx = int(x * template_perc)
    retally = int(y * template_perc)
    x_aux = int(x / 2) - int(retallx / 2)
    y_aux = int(y / 2) - int(retally / 2)

    template = b[y_aux:y_aux + retally, x_aux:x_aux + retallx]
    plt.imshow(cv2.cvtColor(template, cv2.COLOR_BGR2RGB)), plt.title("Template"), plt.xticks([]), plt.yticks([])
    plt.show()

    # #POSADA EN CORRESPONDENCIA, AJUNTAR LES IMATGES (Apply template Matching)#############################################
    '''
    Com a correlaci´o podeu fer servir les que he citat a classe:
    la correlació (basada en convolució en l’espai) = cv2.TM_CCORR, 
    la correlació (basada en producte en l’espai de Fourier) = fftconvolve, 
    la correlació de fase (basada en Fourier) = 'phaseCorr'
    i la correlació normalitzada = NCC (Normalized cross-correlation) = 'cv2.TM_CCORR_NORMED'
    
    Sum of square differences (SSD) =  TM_SQDIFF
    Cross correlation (CC) =  TM_CCORR
    Mean shifted cross correlation (Pearson correlation coefficient) =  TM_CCOEFF
    
    https://stackoverflow.com/questions/58158129/understanding-and-evaluating-template-matching-methods
    '''
    # All the 6 methods for comparison in a list
    methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR', 'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF',
               'cv2.TM_SQDIFF_NORMED', 'phaseCorr', 'convolve2d', 'fftconvolve']

    for meth in methods:
        if meth == 'phaseCorr':
            t0 = time.time()
            r_aux_2 = r.astype('float')
            g_aux_2 = g.astype('float')
            b_aux_2 = b.astype('float')
            #phase correlation basada en fourier
            tuple_shift_g = cv2.phaseCorrelate(g_aux_2, r_aux_2)[0]
            tuple_shift_b = cv2.phaseCorrelate(b_aux_2, r_aux_2)[0]
        
            # Desplaçar la diferencia respecte la vermella i alinear-les
            g_aux = np.roll(g, round(tuple_shift_g[0]), 1)
            g_aux = np.roll(g_aux, round(tuple_shift_g[1]), 0)
        
            b_aux = np.roll(b, round(tuple_shift_b[0]), 1)
            b_aux = np.roll(b_aux, round(tuple_shift_b[1]), 0)
        
            # Resultats
            im_res = np.dstack((r, g_aux, b_aux))
            t1 = time.time()
            print('El mètode ', meth, ' ha tardat: ', t1 - t0, ' segons.')
            plt.imshow(cv2.cvtColor(im_res, cv2.COLOR_BGR2RGB))
            plt.suptitle(meth)
            plt.show()
           
        elif  meth == 'fftconvolve':
            t0 = time.time()
            
            r_aux_2 = r.astype('float')
            g_aux_2 = g.astype('float')
            b_aux_2 = b.astype('float')
            
            mat_r = fftconvolve(r_aux_2, r_aux_2,mode='full')
            mat_g = fftconvolve(g_aux_2, r_aux_2, mode='full')
            mat_b = fftconvolve(b_aux_2, r_aux_2, mode='full')
            
            r_x, r_y = np.unravel_index(mat_r.argmax(), mat_r.shape)
            g_x, g_y = np.unravel_index(mat_g.argmax(), mat_g.shape)
            b_x, b_y = np.unravel_index(mat_b.argmax(), mat_b.shape)
            
            # Desplaçar la diferencia respecte la vermella i alinear-les
            g_aux = np.roll(g, round(r_x - g_x), 1)
            g_aux = np.roll(g_aux, round(r_y - g_y), 0)
        
            b_aux = np.roll(b, round(r_x - b_x), 1)
            b_aux = np.roll(b_aux, round(r_y - b_y), 0)
        
            # Resultats
            im_res = np.dstack((r, g_aux, b_aux))
            t1 = time.time()
            print('El mètode correlació (basada en producte en l’espai de Fourier) ha tardat: ', t1 - t0, ' segons.')
            plt.imshow(cv2.cvtColor(im_res, cv2.COLOR_BGR2RGB))
            plt.suptitle('correlació (basada en producte en l’espai de Fourier)')
            plt.show()
            
        
        elif  meth == 'convolve2d':
            '''
            t0 = time.time()
            r_aux_2 = r.astype('float')
            g_aux_2 = g.astype('float')
            b_aux_2 = b.astype('float')
            
            mat_r = convolve2d(r_aux_2, r,mode='valid')#mode='same')
            mat_g = convolve2d(g_aux_2, r, mode='valid')#mode='same')
            mat_b = convolve2d(b_aux_2, r, mode='valid')#mode='same')
            
            r_x, r_y = np.unravel_index(mat_r.argmax(), mat_r.shape)
            g_x, g_y = np.unravel_index(mat_g.argmax(), mat_g.shape)
            b_x, b_y = np.unravel_index(mat_b.argmax(), mat_b.shape)
            
            # Desplaçar la diferencia respecte la vermella i alinear-les
            g_aux = np.roll(g, round(r_x - g_x), 1)
            g_aux = np.roll(g_aux, round(r_y - g_y), 0)
        
            b_aux = np.roll(b, round(r_x - b_x), 1)
            b_aux = np.roll(b_aux, round(r_y - b_y), 0)
        
            # Resultats
            im_res = np.dstack((r, g_aux, b_aux))
            t1 = time.time()
            print('El mètode correlació (basada en convolució en l’espai) ha tardat: ', t1 - t0, ' segons.')
            plt.imshow(cv2.cvtColor(im_res, cv2.COLOR_BGR2RGB))
            plt.suptitle('correlació (basada en convolució en l’espai)')
            plt.show()
            ''' 
        else:            
            t0 = time.time()
        
            # Obtenir posicions de la cantonada del template
            method = eval(meth)
            res = cv2.matchTemplate(r, template, method)
            min_val_r, max_val_r, min_loc_r, max_loc_r = cv2.minMaxLoc(res)
        
            res2 = cv2.matchTemplate(g, template, method)
            min_val_g, max_val_g, min_loc_g, max_loc_g = cv2.minMaxLoc(res2)
        
            res3 = cv2.matchTemplate(b, template, method)
            min_val_b, max_val_b, min_loc_b, max_loc_b = cv2.minMaxLoc(res3)
        
            # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
            if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
                top_left_r = min_loc_r
                top_left_g = min_loc_g
                top_left_b = min_loc_b
            else:
                top_left_r = max_loc_r
                top_left_g = max_loc_g
                top_left_b = max_loc_b
        
            # Desplaçar la diferencia respecte la vermella i alinear-les
            g_aux = np.roll(g, top_left_r[0] - top_left_g[0], 1)
            g_aux = np.roll(g_aux, top_left_r[1] - top_left_g[1], 0)
        
            b_aux = np.roll(b, top_left_r[0] - top_left_b[0], 1)
            b_aux = np.roll(b_aux, top_left_r[1] - top_left_b[1], 0)
        
            # Resultats
            im_res = np.dstack((r, g_aux, b_aux))
            t1 = time.time()
            print('El mètode ', meth, ' ha tardat: ', t1 - t0, ' segons.')
            plt.imshow(cv2.cvtColor(im_res, cv2.COLOR_BGR2RGB))
            plt.suptitle(meth)
            plt.show()
        
            if meth == 'cv2.TM_SQDIFF_NORMED':
                #Millor imatge amb TM_SQDIFF_NORMED  (millor temps i qualitat)
                cv2.imwrite(filename[0:-4] +'_color.png', im_res)
            
        
            
    

if __name__ == "__main__":
    # execute only if run as a script
    obtainImage('img_5_small.jpg')



