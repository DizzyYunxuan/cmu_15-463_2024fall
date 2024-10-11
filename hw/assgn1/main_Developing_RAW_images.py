import numpy as np
from scipy.interpolate import interp2d
from scipy.signal import convolve2d
from skimage import io
from skimage.color import rgb2gray
from matplotlib import pyplot as plt



def wb_rgb(r, g0, g1, b, rs, gs, bs):

    # Take mean of two green channel. It's damage the details but enough for color checking.

    r = r * rs
    g = (g0 + g1) / 2 * gs
    b = b * bs

    rgb = np.stack([r, g, b], axis=2)
    return rgb

def interp2d_for_tmpRB(ori_pattern, ori_pos):

    # This func only fill the [diagonal] pixels for each raw pixel.
    # The missing two pixel filled by `interp2d_for_G_Or_TmpRb` func.

    h, w = ori_pattern.shape
    x = np.arange(0, w, 1)
    y = np.arange(0, h, 1)
    f = interp2d(x, y, ori_pattern, kind='linear')

    interp_res = np.zeros([h*2, w*2])

    if ori_pos == 0: # r
        index_interp_x = x + 0.5
        index_interp_y = y + 0.5
        interp_res[1::2, 1::2] = f(index_interp_x, index_interp_y)
        interp_res[::2, ::2]   = ori_pattern
    
    if ori_pos == 3: # b
        index_interp_x = x - 0.5
        index_interp_y = y - 0.5
        interp_res[::2, ::2] = f(index_interp_x, index_interp_y)
        interp_res[1::2, 1::2]   = ori_pattern
    
    return interp_res


def interp2d_for_G_Or_TmpRb(ori_pattern, ori_pos):

    # TODO: use interp2d fill the missing pixels
    # This func simply takes the mean of 4 neighbors by convolution

    kernel = np.zeros([3,3])
    kernel[::2, 1::2] = 1
    kernel[1::2, ::2] = 1
    kernel = kernel / 4

    res = convolve2d(ori_pattern, kernel, boundary='symm', mode='same')
    if ori_pos == '03': # r/b
        res[::2, ::2] = ori_pattern[::2, ::2]
        res[1::2, 1::2] = ori_pattern[1::2, 1::2]
    elif ori_pos == '12': # g
        res[1::2, ::2] = ori_pattern[1::2, ::2]
        res[::2, 1::2] = ori_pattern[::2, 1::2]


    return res


# record parameters from dcraw command
black = 150
white = 4095
r_scale = 2.394531
g_scale = 1.000000
b_scale = 1.597656



#--------------------------------------- Python initials -----------------------------------------------------------------#
raw = io.imread('hw/assgn1/data/campus.tiff')
print(raw.shape)
print(raw.dtype)
raw = raw.astype(np.float32)

#--------------------------------------- Linearization -------------------------------------------------------------------#
raw_linear = (raw - black) / (white - black)
raw_linear = np.clip(raw_linear, 0, 1)


#--------------------------------------- Identifying the correct Bayer pattern -------------------------------------------#
# if grbg
g0 = raw_linear[::2, ::2]
r  = raw_linear[::2, 1::2]
b  = raw_linear[1::2, ::2]
g1 = raw_linear[1::2, 1::2]
rgb = wb_rgb(r, g0, g1, b, r_scale, g_scale, b_scale)
io.imsave('hw/assgn1/experiments/grbg.png', np.uint8(np.clip(rgb, 0, 1)*255))

# if rggb
r  = raw_linear[::2, ::2]
g0 = raw_linear[::2, 1::2]
g1 = raw_linear[1::2, ::2]
b  = raw_linear[1::2, 1::2]
rgb = wb_rgb(r, g0, g1, b, r_scale, g_scale, b_scale)
io.imsave('hw/assgn1/experiments/rggb.png', np.uint8(np.clip(rgb, 0, 1)*255))

# if bggr
b  = raw_linear[::2, ::2]
g0 = raw_linear[::2, 1::2]
g1 = raw_linear[1::2, ::2]
r  = raw_linear[1::2, 1::2]
rgb = wb_rgb(r, g0, g1, b, r_scale, g_scale, b_scale)
io.imsave('hw/assgn1/experiments/bggr.png', np.uint8(np.clip(rgb, 0, 1)*255))

# if gbrg
g0 = raw_linear[::2, ::2]
b  = raw_linear[::2, 1::2]
r  = raw_linear[1::2, ::2]
g1 = raw_linear[1::2, 1::2]
rgb = wb_rgb(r, g0, g1, b, r_scale, g_scale, b_scale)
io.imsave('hw/assgn1/experiments/gbrg.png', np.uint8(np.clip(rgb, 0, 1)*255))


# #--------------------------------------- White balancing [rggb seems the correct pattern] -------------------------------------------#
r  = raw_linear[::2, ::2]
g0 = raw_linear[::2, 1::2]
g1 = raw_linear[1::2, ::2]
b  = raw_linear[1::2, 1::2]


# gray world assumption
r_mean = np.mean(r)
g_mean = np.mean((g0 + g1)/2)
b_mean = np.mean(b)

rgb_grayAs = wb_rgb(r, g0, g1, b, g_mean / r_mean, 1.0, g_mean / b_mean)
io.imsave('hw/assgn1/experiments/rgb_grayAs.png', np.uint8(np.clip(rgb_grayAs, 0, 1)*255))

# white world assumption
r_max = np.max(r)
g_max = np.max((g0 + g1)/2)
b_max = np.max(b)

rgb_whiteAs = wb_rgb(r, g0, g1, b, g_max / r_max, 1.0, g_max / b_max)
io.imsave('hw/assgn1/experiments/rgb_whiteAs.png', np.uint8(np.clip(rgb_whiteAs, 0, 1)*255))

# preseted white balance 
rgb_preSetWb = wb_rgb(r, g0, g1, b, r_scale, g_scale, b_scale)
io.imsave('hw/assgn1/experiments/rgb_preSetWb.png', np.uint8(np.clip(rgb_preSetWb, 0, 1)*255))


# #--------------------------------------- Demosaicing [preset white balance seems better] -------------------------------------------#
raw_linear_wb = np.copy(raw_linear)
raw_linear_wb[::2, ::2]   *= r_scale
raw_linear_wb[::2, 1::2]  *= g_scale
raw_linear_wb[1::2, ::2]  *= g_scale
raw_linear_wb[1::2, 1::2] *= b_scale

r_ch = interp2d_for_tmpRB(raw_linear_wb[ ::2,  ::2], 0)
g_ch = np.copy(raw_linear_wb)
g_ch[::2, ::2] = 0
g_ch[1::2, 1::2] = 0
b_ch = interp2d_for_tmpRB(raw_linear_wb[1::2, 1::2], 3)

r_ch = interp2d_for_G_Or_TmpRb(r_ch, '03')
g_ch = interp2d_for_G_Or_TmpRb(g_ch, '12')
b_ch = interp2d_for_G_Or_TmpRb(b_ch, '03')

rgb_dmsc = np.stack([r_ch, g_ch, b_ch], axis=2)
io.imsave('hw/assgn1/experiments/rgb_dmsc.png', np.uint8(np.clip(rgb_dmsc, 0, 1)*255))




#--------------------------------------- Color space correction -------------------------------------------#
# dcraw shows that the image was captured by Nikon D3400, so I found the Nikon D3400 Mxyz->cam in dcraw.c
# Mxyz_cam = 6988,-1384,-714,-5631,13410,2447,-1485,2204,7318

Mxyz_cam = np.array([6988,-1384,-714,-5631, 13410, 2447,-1485, 2204, 7318])
Mxyz_cam = Mxyz_cam.reshape([3,3]) / 10000
MsRGB_XYZ = np.array([[0.4124564, 0.3575761, 0.1804375],
                      [0.2126729, 0.7151522, 0.0721750],
                      [0.0193339, 0.1191920, 0.9503041]])

MsRGB_cam = np.matmul(Mxyz_cam, MsRGB_XYZ)
norm_row_coe = np.sum(MsRGB_cam, axis=1, keepdims=True)
MsRGB_cam = MsRGB_cam / norm_row_coe
MsRGB_cam = np.matrix(MsRGB_cam)

h, w, c = rgb_dmsc.shape
cam_dmsc_rs = rgb_dmsc.reshape([-1, 3]).T
srgb_dmsc_rs = np.matmul(MsRGB_cam.I, cam_dmsc_rs)
srgb_dmsc_rs = np.array(srgb_dmsc_rs).T
srgb_dmsc = srgb_dmsc_rs.reshape([h, w, 3])
io.imsave('hw/assgn1/experiments/srgb.png', np.uint8(np.clip(srgb_dmsc, 0, 1)*255))


#------------------------------ Brightness adjustment and gamma encoding ---------------------------------#
# scaling mean gray intensity
gray = rgb2gray(srgb_dmsc)
for scale in np.arange(1.0, 3.1, 0.5):
    gray_scaled = gray * scale
    gray_scaled = np.clip(gray_scaled, 0, 1)
    io.imsave('hw/assgn1/experiments/gray_scaled_{:.2f}.png'.format(scale), np.uint8(np.clip(gray_scaled, 0, 1)*255))


# gamma correction
srgb_non_linear = np.copy(srgb_dmsc)
T = 0.0031308
srgb_non_linear[srgb_dmsc<=T] = srgb_dmsc[srgb_dmsc<=T] * 12.92
srgb_non_linear[srgb_dmsc>T]  = (1 + 0.055) * srgb_dmsc[srgb_dmsc>T] ** (1/2.4) - 0.055
io.imsave('hw/assgn1/experiments/srgb_non_linear.png', np.uint8(np.clip(srgb_non_linear, 0, 1)*255))
    


#------------------------------------------ Compression ---------------------------------------------------#
# save png
io.imsave('hw/assgn1/experiments/srgb_non_linear.png', np.uint8(np.clip(srgb_non_linear, 0, 1)*255))

# save jpeg
io.imsave('hw/assgn1/experiments/srgb_non_linear_q95.jpeg', np.uint8(np.clip(srgb_non_linear, 0, 1)*255), quality=95)
io.imsave('hw/assgn1/experiments/srgb_non_linear_q75.jpeg', np.uint8(np.clip(srgb_non_linear, 0, 1)*255), quality=75)
io.imsave('hw/assgn1/experiments/srgb_non_linear_q55.jpeg', np.uint8(np.clip(srgb_non_linear, 0, 1)*255), quality=55)
io.imsave('hw/assgn1/experiments/srgb_non_linear_q35.jpeg', np.uint8(np.clip(srgb_non_linear, 0, 1)*255), quality=35)
io.imsave('hw/assgn1/experiments/srgb_non_linear_q15.jpeg', np.uint8(np.clip(srgb_non_linear, 0, 1)*255), quality=15)

# The quality=75 is better.



#------------------------------------------ Perform manual white balancing ---------------------------------------------------#
raw_linear_wb = np.copy(raw_linear)

r_ch = interp2d_for_tmpRB(raw_linear_wb[ ::2,  ::2], 0)
g_ch = np.copy(raw_linear_wb)
g_ch[::2, ::2] = 0
g_ch[1::2, 1::2] = 0
b_ch = interp2d_for_tmpRB(raw_linear_wb[1::2, 1::2], 3)

r_ch = interp2d_for_G_Or_TmpRb(r_ch, '03')
g_ch = interp2d_for_G_Or_TmpRb(g_ch, '12')
b_ch = interp2d_for_G_Or_TmpRb(b_ch, '03')

rgb_dmsc = np.stack([r_ch, g_ch, b_ch], axis=2)
io.imsave('hw/assgn1/experiments/rgb_dmsc_noWB.png', np.uint8(np.clip(rgb_dmsc, 0, 1)*255))

plt.imshow(rgb_dmsc)
left_top, right_bot = plt.ginput(2)
left_top = list(map(int, left_top))
right_bot = list(map(int, right_bot))
print(left_top)
print(right_bot)

# h, w patch
awb_patch = rgb_dmsc[left_top[1]:right_bot[1], left_top[0]:right_bot[0], :]

# target green channel
green_ch = awb_patch[:,:,1]
green_ch_vec = np.matrix(green_ch.reshape([-1, 1]))

# Least Squares Method
r_ch_vec = np.matrix(awb_patch[:,:,0].reshape([-1, 1]))
r_scale = (r_ch_vec.T * r_ch_vec).I * r_ch_vec.T * green_ch_vec

b_ch_vec = np.matrix(awb_patch[:,:,2].reshape([-1, 1]))
b_scale = (b_ch_vec.T * b_ch_vec).I * b_ch_vec.T * green_ch_vec

rgb_dmsc_wb = np.copy(rgb_dmsc)
rgb_dmsc_wb[:,:,0] *= r_scale
rgb_dmsc_wb[:,:,2] *= b_scale
io.imsave('hw/assgn1/experiments/rgb_dmsc_WB_patchCalWb_{}-{}_{}-{}.png'.format(*left_top, *right_bot), np.uint8(np.clip(rgb_dmsc_wb, 0, 1)*255))



# Learn to use dcraw #
#  ./hw/assgn1/utils/dcraw -e hw/assgn1/data/campus.nef
#