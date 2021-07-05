# Processamento de Imagens
# Trabalho 1
# Caio Ferreira Bernardo - NUSP 9276936
# João Pedro Hannauer - NUSP 9390486



import numpy as np
import imageio as io
from scipy.ndimage import convolve


#generating the normalized hist
def norm_hist(img, bins, b):
    result =  np.histogram(img, bins = np.power(2,b))[0]
    result = result/np.sum(result)
    return result/np.linalg.norm(result), result

#img qunatization
def quantization(img, b):

    return np.right_shift(img.astype(np.uint8), 8-b)
#luminance function
def luminance(img):
    return np.floor(0.299*img[:,:,0] + 0.587*img[:,:,1] + 0.114*img[:,:,2])

#calculatin the co ocurrence matrix
def intensity_co_ocurrence(img,d_x,d_y, levels):
    srcdata = img.copy()
    ret=np.zeros((levels, levels))
   
    (height,width) = srcdata.shape
    


    for j in range(height-d_y):
        for i in range(width-d_x):
            rows = srcdata[j][i]
            cols = srcdata[j + d_y][i+d_x]
            ret[rows][cols]+=1.0
    

    return ret/np.sum(ret)

#calculating the texture features
def feature_computer(matrix, levels, eps = 0.001):
    con = 0.0
    ent = 0.0
    energy = 0.0
    homo = 0.0
    corr = 0
    
    result = []
    
    energy = np.sum(np.power(matrix, 2))
    
    ent = np.sum(matrix*np.log(matrix + eps))
    
    i=j=np.arange(levels)
    xv, yv = np.meshgrid(i,j, sparse=False, indexing='xy') 
    con = np.sum(np.power((xv-yv), 2) * matrix)
    homo = np.sum(matrix/(1 + np.absolute(xv - yv)))
    
    mu_i = calc_mean_x(matrix, levels)
    mu_j = calc_mean_y(matrix, levels)
    
    std_i = calc_std_x(matrix, mu_i, levels)
    std_j = calc_std_y(matrix, mu_j, levels)
   
    corr = (np.sum((xv * yv) * matrix) - mu_i * mu_j)/(std_i * std_j)
  
    
    con = (1/(np.power(levels, 2)))*con
    result.append(energy)
    result.append(-ent)
    result.append(con)
    result.append(corr)
    result.append(homo)
    
    result = np.array(result)
    return result/np.linalg.norm(result), result

#calc the vertical mean
def calc_mean_x(matrix,  size):
    ui = 0
    for i in range(size):
        for j in range(size):
            ui += i*matrix[i][j]
    return ui

#calc the horizontal mean
def calc_mean_y(matrix,  size):
    uj = 0
    temp = 0
    for j in range(size):
        for i in range(size):
            uj += j*matrix[i][j]
    return uj

#calc the vertical mean
def calc_std_x(matrix, ui, size):
    aux = 0
    std_x = 0
    for i in range(size):
        for j in range(size):
            std_x += np.power((i-ui), 2)*matrix[i][j]
            
    return std_x

#calc the horizontal mean
def calc_std_y(matrix, uj, size):
    aux = 0
    std_y = 0
    for j in range(size):
        for i in range(size):
            std_y += np.power((j-uj), 2)*matrix[i][j]
            
    return std_y


#calc the gradients
def calc_grads(matrix):
    weights_sobel_x =  np.array([[-1,-2,-1], [0,0,0], [1,2,1]])
    weights_sobel_y = np.array([[-1,0,1], [-2,0,2], [-1, 0,1]])
    
    grad_x = convolve(matrix, weights_sobel_x)
    grad_y = convolve(matrix, weights_sobel_y)
    
    return grad_x, grad_y



#calc the magnitude
def gradient_magnitude(horizontal_gradient, vertical_gradient):
    horizontal_gradient_square = np.power(horizontal_gradient, 2)
    vertical_gradient_square = np.power(vertical_gradient, 2)
    sum_squares = horizontal_gradient_square + vertical_gradient_square
    grad_magnitude = np.sqrt(sum_squares)/np.sum(np.sqrt(sum_squares))

    return grad_magnitude


#calc the angles
def gradient_direction(horizontal_gradient, vertical_gradient):
    grad_direction = np.arctan(horizontal_gradient/(vertical_gradient))
    
    grad_direction = grad_direction + (np.pi/2)
    
    grad_direction = np.degrees(grad_direction)
    
    return grad_direction

#counting the bins
def gradient_bins_counts(gradient_dir):
    row, col = gradient_dir.shape 
    result = np.zeros(shape = gradient_dir.shape)

    for i in range(row):
        for j in range(col):
           
            if gradient_dir[i][j] >=0 and gradient_dir[i][j] < 20:
                result[i][j] = 0
            elif gradient_dir[i][j] >=20 and gradient_dir[i][j] < 40:
                    result[i][j] = 1
            elif gradient_dir[i][j] >=40 and gradient_dir[i][j] < 60:
                    result[i][j] = 2
            elif gradient_dir[i][j] >=60 and gradient_dir[i][j] < 80:
                    result[i][j] = 3
            elif gradient_dir[i][j] >=80 and gradient_dir[i][j] < 100:
                    result[i][j] = 4
            elif gradient_dir[i][j] >=100 and gradient_dir[i][j] < 120:
                    result[i][j] = 5
            elif gradient_dir[i][j] >=120 and gradient_dir[i][j] < 140:
                    result[i][j] = 6
            elif gradient_dir[i][j] >=140 and gradient_dir[i][j] < 160:
                    result[i][j] = 7
            elif gradient_dir[i][j] >=160 :
                result[i][j] = 8
                
    result = result.astype(int)
    return result


#calculating the agg magnitude
def calc_agg_mag(bins_counts, gradient_mags):
    result = np.zeros(9)
    row, col = bins_counts.shape
    
    for i in range(row):
        for j in range(col):
            aux =  bins_counts[i][j]
            result[aux] = result[aux] + gradient_mags[i][j]
            
    return result/np.linalg.norm(result), result

#just a test function
def get_windows(arr, window_size=64, step=32):
    windows = []
    row = 0
    col = 0
    max_row, max_col = arr.shape
    while row +step < max_row:
        while col + step < max_col:
            windows.append(arr[row:row+window_size, col:col+window_size])
            col += step
        row += step
        col = 0
    
    return windows

#defining the rmse
def RMSE(g, R):
    
    
    # Convert the images to int32 allowing negative values to compute RMSE
    return np.sqrt(np.sum(np.power((g.astype(float)-R.astype(float)),2)))

#looking for the object 
def find_img(path1, path2, quantization_param, window_size = 32, step=16):
    
    errors = []
    img_big = io.imread(path2)
    img_small = io.imread(path1)
    
    img_big_bw = luminance(img_big)
    img_big_bw_quat = quantization(img_big_bw, quantization_param)
    img_big_levels = np.max(img_big_bw_quat)
    
    img_small_bw = luminance(img_small)
    img_small_bw_quat = quantization(img_small_bw, quantization_param)
    img_small_levels = np.max(img_small_bw_quat)
    
    
    d_c, dc_check = norm_hist(img_small_bw_quat, bins=img_small_levels, b=quantization_param)
    
    co_ocurrence_m = intensity_co_ocurrence(img_small_bw_quat, 1, 1,img_small_levels+1)
    
    d_t, dt_check = feature_computer(co_ocurrence_m, img_small_levels+1)
    
    
    gx, gy = calc_grads(img_small_bw_quat.astype(float))
    
    g_mag = gradient_magnitude(gy, gx)
    g_dir = gradient_direction(gy, gx)
   
    
    
    bins_count = gradient_bins_counts(g_dir)
    
    d_g, dg_check = calc_agg_mag(bins_count, g_mag)
    
    small_concat = np.concatenate((d_c, d_t, d_g))
    
    error = np.inf
    
    row = 0
    col = 0
    max_row, max_col = img_big_bw_quat.shape
    window_i, window_j = 0, 0
    while (row + step) < max_row:
        while (col + step)< max_col:
            window = (img_big_bw_quat[row:row+window_size, col:col+window_size])

            
            d_c_window, _= norm_hist(window, bins = img_big_levels, b= quantization_param)
    
            co_ocurrence_m = intensity_co_ocurrence(window, 1, 1,img_big_levels+1)
            d_t_window, _ = feature_computer(co_ocurrence_m, img_big_levels+1)


            gx, gy = calc_grads(window.astype(float))
            g_mag = gradient_magnitude(gy, gx)
            g_dir = gradient_direction(gy, gx)

            bins_count = gradient_bins_counts(g_dir)
            d_g_window, _ = calc_agg_mag(bins_count, g_mag)
            
            window_concat = np.concatenate((d_c_window, d_t_window, d_g_window))
            
            
            erro_temp = RMSE(small_concat, window_concat)
            if erro_temp < error:
                
                result = (window_i, window_j)
                error = erro_temp
               
            
            col += step
            window_j += 1
        
        window_i += 1
        row += step
        col = 0
        window_j = 0
    
    #uncoment if want to check descriptos values before and after normalization

    # print(f'd_c: {d_c}')
    # print(f'd_t: {d_t}')
    # print(f'd_g: {d_g}')

    # print(f'\nd_c_no_norm: {dc_check}')
    # print(f'd_t_no_norm: {dt_check}')
    # print(f'd_g_no_norm: {dg_check}')
    return result


def main():
    #turning off the warnings
    np.seterr(divide='ignore', invalid='ignore')
    
    # Get user inputs
    np.set_printoptions(suppress=True)
    filename_small = input().rstrip()
    filename_bigger = input().rstrip()
    b = int(input().rstrip())

    coordinates = find_img(filename_small, filename_bigger, b)
    print(f'{coordinates[0]} {coordinates[1]}')


if __name__ == "__main__":
	main()
