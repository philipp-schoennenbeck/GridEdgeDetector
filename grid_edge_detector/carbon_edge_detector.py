
from matplotlib import image
import numpy as np
import mrcfile
from skimage.draw import disk
from scipy.signal import convolve
from pathlib import Path
from PIL import Image
from matplotlib import pyplot as plt


# def mask_carbon_edge_per_file(path, gridsizes, cut_off, ps, circles=None, idx=None, get_hist_data=False, to_resize=False, resize=7):
#     if isinstance(path, (str, Path)):
#         path = Path(path)
#         if path.suffix in [".mrc", ".MRC", ".rec", ".REC"]:
#             image = mrcfile.open(path, permissive=True).data*1
#         else:
#             image = np.array(Image.open(path).convert("L"))
#     else:
#         image = path
#     data = image - np.mean(image)
#     data /= np.std(data)
    
#     if to_resize:
#         original_ps = ps
#         original_shape = data.shape
#         ratio = ps / resize
#         new_shape = None
#         if ratio < 1:
#             new_shape = [int(os * ratio) for os in original_shape]
#             data = np.array(Image.fromarray(data).resize(new_shape[::-1]))
#             ps = resize

#     differences = []
#     coords = []
#     # convolved_images = []
#     # convolved_images_neg = []
#     if get_hist_data:
#         hist_data = {}
#     for gridsize in gridsizes:


        
#         if circles is None:
#             radius = int(gridsize/ps // 2)
#             circle_coords = disk((radius,radius), radius)
#             circle = np.zeros((radius*2, radius*2))
#             circle[circle_coords[0], circle_coords[1]] = 1
#             circle = circle/np.sum(circle)
#             neg_circle = (circle==0)*1
#             neg_circle = neg_circle / np.sum(neg_circle)
#             combined_circle = circle - neg_circle
#             diff_image = convolve(data, combined_circle, "full", method="fft")
#             # convolved_image_neg = convolve(data, neg_circle, "full", method="fft")
#         else:
#             convolved_image = convolve(data, circles[gridsize][0], "full", method="fft")
#             convolved_image_neg = convolve(data, circles[gridsize][1], "full", method="fft")
#             diff_image = convolved_image - convolved_image_neg
        
#         coord = np.argmax(diff_image)
#         coord = np.unravel_index(coord, diff_image.shape)
#         if circles is None:
#             coord = coord[0] - circle.shape[0]//2, coord[1] - circle.shape[1]//2
#         else:
#             coord = coord[0] - circles[gridsize][0].shape[0]//2, coord[1] - circles[gridsize][0].shape[1]//2

#         differences.append(np.max(diff_image))
#         if to_resize and ratio < 1:
#             coord = (np.array(coord) / ratio).astype(np.int32)
#         coords.append(coord)
#         # convolved_images.append(convolved_image)
#         # convolved_images_neg.append(convolved_image_neg)

#         if get_hist_data:
#             values, edges = np.histogram(diff_image, bins=50)
            
#             hist_data[gridsize] = {"values":values, "edges":edges, "threshold":cut_off, "center":coord}

#     argmax = np.argmax(differences)
#     coord = coords[argmax]
#     if to_resize and ratio < 1:
#         ps = original_ps

#     mask = np.zeros_like(image, dtype=np.uint8)
#     if np.max(differences ) > cut_off:
#         yy,xx = disk(coord, gridsizes[np.argmax(differences)]/ps // 2, shape=image.shape,)
#         mask[yy,xx] = 1
#     else:
#         mask = np.ones_like(image, dtype=np.uint8)
#     # if idx is not None:
#     #     output = Path("/Data/erc-3/schoennen/membrane_analysis_toolkit/test_code/output_for_progress_report_20221112")
#     #     plt.imsave(output / f"{idx}_original.png",image, cmap="gray")
#     #     plt.imsave(output / f"{idx}_mask.png", mask, vmin=0, vmax=1, cmap="gray")
#     #     plt.imsave(output / f"{idx}_conv.png",convolved_image, cmap="gray")
#     #     plt.imsave(output / f"{idx}_conv_neg.png",convolved_image_neg, cmap="gray")
#     #     plt.imsave(output / f"{idx}_diff_image.png",diff_image, cmap="gray")
#     if get_hist_data:
#         return mask, hist_data, gridsizes[np.argmax(differences)]
#     return mask


def mask_carbon_edge_per_file(path, gridsizes, cut_off, ps, circles=None, idx=None, get_hist_data=False, to_resize=False, resize=7):
    if isinstance(path, (str, Path)):
        path = Path(path)
        if path.suffix in [".mrc", ".MRC", ".rec", ".REC"]:
            image = mrcfile.open(path, permissive=True).data*1
        else:
            image = np.array(Image.open(path).convert("L"))
    else:
        image = path
    data = image - np.mean(image)
    data /= np.std(data)
    
    if to_resize:
        original_ps = ps
        original_shape = data.shape
        ratio = ps / resize
        new_shape = None
        if ratio < 1:
            new_shape = [int(os * ratio) for os in original_shape]
            data = np.array(Image.fromarray(data).resize(new_shape[::-1]))
            ps = resize

    differences = []
    coords = []
    # convolved_images = []
    # convolved_images_neg = []
    if get_hist_data:
        hist_data = {}
    for gridsize in gridsizes:


        
        if circles is None:
            radius = int(gridsize/ps // 2)
            circle_coords = disk((radius,radius), radius)
            
            circle = np.zeros((radius*2, radius*2))
            circle[circle_coords[0], circle_coords[1]] = 1
            circle = circle/np.sum(circle)
            neg_circle = (circle==0)*1
            neg_circle = neg_circle / np.sum(neg_circle)
            combined_circle = circle - neg_circle
            diff_image = convolve(data, combined_circle, "full", method="fft")
            # convolved_image_neg = convolve(data, neg_circle, "full", method="fft")
        else:
            convolved_image = convolve(data, circles[gridsize][0], "full", method="fft")
            convolved_image_neg = convolve(data, circles[gridsize][1], "full", method="fft")
            diff_image = convolved_image - convolved_image_neg
        
        coord = np.argmax(diff_image)
        coord = np.unravel_index(coord, diff_image.shape)
        if circles is None:
            coord = coord[0] - circle.shape[0]//2, coord[1] - circle.shape[1]//2
        else:
            coord = coord[0] - circles[gridsize][0].shape[0]//2, coord[1] - circles[gridsize][0].shape[1]//2

        differences.append(np.max(diff_image))
        if to_resize and ratio < 1:
            coord = (np.array(coord) / ratio).astype(np.int32)
        coords.append(coord)
        # convolved_images.append(convolved_image)
        # convolved_images_neg.append(convolved_image_neg)

        if get_hist_data:
            values, edges = np.histogram(diff_image, bins=50)
            
            hist_data[gridsize] = {"values":values, "edges":edges, "threshold":cut_off, "center":coord}

    argmax = np.argmax(differences)
    coord = coords[argmax]
    if to_resize and ratio < 1:
        ps = original_ps

    mask = np.zeros_like(image, dtype=np.uint8)
    if np.max(differences ) > cut_off:
        yy,xx = disk(coord, gridsizes[np.argmax(differences)]/ps // 2, shape=image.shape,)
        mask[yy,xx] = 1
    else:
        mask = np.ones_like(image, dtype=np.uint8)

    if get_hist_data:
        return mask, hist_data, gridsizes[np.argmax(differences)]
    return mask


def mask_carbon_edge(paths_or_images, gridsizes=[11000,11500,12000,12500, 13000,19000,19500,20000,20500,21000], cut_off=0.02, ps=7, njobs=1):
    import multiprocessing as mp
    circles = {}
    for gridsize in gridsizes:
        radius = int(gridsize/ps // 2)
        circle_coords = disk((radius,radius), radius)
        circle = np.zeros((radius*2, radius*2))
        circle[circle_coords[0], circle_coords[1]] = 1
        circle = circle/np.sum(circle)
        neg_circle = (circle==0)*1
        neg_circle = neg_circle / np.sum(neg_circle)
        circles[gridsize] = [circle, neg_circle]

    circle_coords = {gridsize:disk((int(gridsize/ps // 2), int(gridsize/ps // 2)), int(gridsize/ps // 2)) for gridsize in gridsizes}
    # 

    if njobs > 1:
        with mp.Pool(njobs) as pool:
            results = [pool.apply_async(mask_carbon_edge_per_file, [path, gridsizes, cut_off, ps, circles, idx]) for idx, path in enumerate(paths_or_images)]
            masks = [res.get() for res in results]
    else:

        masks = [mask_carbon_edge_per_file(path, gridsizes, cut_off, ps, circles) for path in paths_or_images]
    
        
    return masks
    #
 
if __name__ == "__main__":
    pass