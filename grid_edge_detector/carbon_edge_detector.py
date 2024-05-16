
import numpy as np
import mrcfile
from skimage.draw import disk
from scipy.signal import convolve
from scipy.ndimage import convolve as nd_convolve
from pathlib import Path
from PIL import Image
from matplotlib import pyplot as plt
import sparse
from PIL import Image, ImageOps
# from matplotlib import pyplot as plt


 

def gauss(fx,fy,sig):
    r = np.fft.fftshift(np.sqrt(fx**2 + fy**2))
    res = -2*np.pi**2*(r*sig)**2

    return np.exp(-2*np.pi**2*(r*sig)**2)

def gaussian_filter(im,sig,apix):
    '''
        sig (real space) and apix in angstrom
    '''
    sig = sig/2/np.pi
    fx,fy = np.meshgrid(np.fft.fftfreq(im.shape[1],apix),\
                        np.fft.fftfreq(im.shape[0],apix))

    im_fft = np.fft.fftshift(np.fft.fft2(im))
    fil = gauss(fx,fy,sig*apix)
    
    im_fft_filtered = im_fft*fil
    newim = np.real(np.fft.ifft2(np.fft.ifftshift(im_fft_filtered)))
    
    return newim



def load_mrc(file):
    with mrcfile.open(file, permissive=True) as f:
        data = f.data * 1
        pixel_size = f.voxel_size["x"]
    return data, pixel_size

def load_jpg_png(file):
    img = ImageOps.grayscale(Image.open(file))
    return np.array(img)

def load_npz(file):
    data = sparse.load_npz(file).todense()
    
    if data.ndim == 3:
        lowest_axes = np.argmin(data.shape)
        data = np.moveaxis(data, lowest_axes, [0])
    return data

suffix_function_dict = {
    ".mrc":load_mrc,
    ".png":load_jpg_png,
    ".jpg":load_jpg_png,
    ".jpeg":load_jpg_png,
    ".npz":load_npz,
    ".rec":load_mrc
}

def load_file(file):
    global suffix_function_dict
    suffix = Path(file).suffix.lower()
    if suffix in suffix_function_dict:
        data = suffix_function_dict[suffix](file)
        if not isinstance(data, tuple):
            data = (data, None)
        return data 
    else:
        raise ValueError(f"Could not find a function to open a {suffix} file.")




def test_new_stuff(image, radius, pixel_size):
    radius = int(radius)
    y, x = np.ogrid[-radius: radius + 1, -radius: radius + 1]
    circular_mask = x**2 + y**2 <= radius**2

    # Normalize the mask to have values between 0 and 1
    # circular_mask = circular_mask.astype(float) / circular_mask.sum()
    circular_mask = circular_mask * 1

    # Pad the image to handle border cases
    padded_image = np.pad(image, radius, mode='constant', constant_values=0)
    value_image = np.ones_like(image)
    value_image = np.pad(value_image, radius, mode='constant', constant_values=0)
    # Convolve the image with the circular mask


    convolved_image = convolve(padded_image, circular_mask,"same",)
    inside_size = convolve(value_image, circular_mask, "same",)


    sum_total = np.sum(image)
    size = image.size
    outside_size = size - inside_size
    outside_values = sum_total - convolved_image

    mean_difference = np.zeros_like(convolved_image)
    mask = np.logical_and(inside_size > 100, outside_size > 100)

    mean_difference[mask] = np.abs(convolved_image[mask] / inside_size[mask] - outside_values[mask]/ outside_size[mask]) 

    return mean_difference






def create_difference_map(radius, detect_ring, ring_width, to_size, new_data, coverage_percentage, outside_coverage_percentage):
    y, x = np.ogrid[-radius: radius + 1, -radius: radius + 1]
    if not detect_ring:
        circular_mask = x**2 + y**2 <= radius**2
        threshold_mask = circular_mask
    else:
        circular_mask = np.logical_and(x**2 + y**2 <= (radius + ring_width/2/to_size)**2,x**2 + y**2 >= (radius - ring_width/2 / to_size)**2)
        threshold_mask = x**2 + y**2 <= radius**2



    circular_mask = circular_mask * 1

    # Pad the image to handle border cases
    padded_image = np.pad(new_data, radius, mode='constant', constant_values=0)
    value_image = np.ones_like(new_data)
    value_image = np.pad(value_image, radius, mode='constant', constant_values=0)
    # Convolve the image with the circular mask


    convolved_image = convolve(padded_image, circular_mask,"same",)
    inside_size = convolve(value_image, circular_mask, "same",)
    threshold_image = convolve(value_image, threshold_mask, "same" )


    sum_total = np.sum(new_data)
    size = new_data.size
    outside_size = size - inside_size
    outside_values = sum_total - convolved_image

    output = np.zeros_like(convolved_image)
    mask = np.logical_and(threshold_image > size * coverage_percentage, size - threshold_image > size * outside_coverage_percentage)

    output[mask] = np.abs(convolved_image[mask] / inside_size[mask] - outside_values[mask]/ outside_size[mask])
    return output



def find_grid_hole_per_file(current_file, to_size=100, diameter=[12000], threshold=0.005, coverage_percentage=0.5, outside_percentage=1, outside_coverage_percentage=0.05,
                             detect_ring=True, ring_width=1000, return_hist_data=False, pixel_size=None, wobble=0, high_pass=1500,crop=50, distance=0, return_ring_width=0, **kwargs):
    # TODO: fix wobble, faster disk by circular mask, strange low thingy (removes radius padding)
    from datetime import datetime
    now = datetime.now()

    return_results = None
    hist_data = {}
    current_file = Path(current_file)
    if current_file.suffix != ".mrc" and pixel_size is None:
        raise TypeError(f"{current_file} is not mrc file and not pixel size was given")
    if not isinstance(diameter, (list, tuple)):
        diameter = [diameter]
    for r in diameter:


        radius = int(r/to_size // 2)  
            
        data, mrc_ps = load_file(current_file)

        if pixel_size is None:
            ps = mrc_ps
        else:
            ps = pixel_size

        # Clip data
        middle = np.median(data)
        std = np.std(data)
        left = middle - std * 4
        right = middle + std * 4
        data = np.clip(data, left, right)

        # Normalize data
        data = data -np.min(data)
        data = data / np.max(data)


        # Reshape image
        ratio = ps / to_size
        new_shape = [round(s * ratio) for s in data.shape]
        new_data = np.array(Image.fromarray(data).resize(new_shape[::-1]))

        
        if not np.isclose(high_pass, 0):
            sig = int(high_pass / to_size)
            sig += (sig + 1) % 2
            new_data = gaussian_filter(new_data,0,to_size) - gaussian_filter(new_data,sig,to_size)

        if crop > 0:
            cut_off = max(1, int(crop * ratio))
            new_data = new_data[cut_off:-cut_off, cut_off:-cut_off]
        else:
            cut_off = 0
        
        # print(f"Preparing took {(datetime.now() - now).total_seconds()}s")
        now = datetime.now()
        output = create_difference_map(radius, detect_ring, ring_width, to_size, new_data, coverage_percentage, outside_coverage_percentage)
        # print(f"Difference map took {(datetime.now() - now).total_seconds()}s")
        now = datetime.now()
        low = int(-radius)

        highest_value = np.max(output)


        
        

        wobble_steps = 0.0025
        
        if wobble>0:
            seen_radii = set()
            start = (1-wobble) * r/to_size / 2
            end = (1+wobble) * r/to_size / 2
            wobble_radii = np.arange(start=start, stop=end, step=wobble_steps * r / to_size / 2, )
            wobble_radii = np.concatenate((wobble_radii, [end]))
            
            
            for wobble_r in wobble_radii:
                
                wobble_r = int(wobble_r)
                if wobble_r in seen_radii:
                    continue
                seen_radii.add(wobble_r)
                new_output = create_difference_map(wobble_r, detect_ring, ring_width, to_size, new_data, coverage_percentage, outside_coverage_percentage)
                current_max = np.max(new_output)
                if current_max > highest_value:
                    highest_value = current_max 
                    output = new_output
                    r = wobble_r * to_size * 2



        circle_center = np.unravel_index(np.argmax(output), output.shape)


        orig_center = ( np.array(circle_center) + cut_off + low ) / (ps/to_size)
        orig_y, orig_x = disk(orig_center, r/ps / 2, shape=data.shape)
        orig_mask = np.zeros_like(data, dtype=np.uint8)
        orig_mask[orig_y, orig_x] = 1

        result_mask = np.zeros_like(data, dtype=np.uint8)
        if np.max(output) > threshold:
            result_mask[orig_y, orig_x] = 1



        values, edges = np.histogram(output, bins=50)

        hist_data[r] = {"values":values, "edges":edges, "threshold":threshold, "center":orig_center}
        if return_results is None:
            if np.abs(distance) > 0 and np.abs(distance) < r / 2:
                orig_y, orig_x = disk(orig_center, np.abs(distance)/ps, shape=data.shape)
                if distance > 0:
                    result_mask[orig_y, orig_x] = 0
                else:
                    result_mask = np.zeros_like(result_mask)
                    result_mask[orig_y, orig_x] = 1
            return_results = {"mask":result_mask, "max":np.max(output), "r":r}
            
            result_output = output
            result_new_data = new_data
        else:
            if np.max(output) > return_results["max"]:
                if np.abs(distance) > 0 and np.abs(distance) < r / 2:
                    orig_y, orig_x = disk(orig_center, np.abs(distance)/ps, shape=data.shape)
                    if distance > 0:
                        result_mask[orig_y, orig_x] = 0
                    else:
                        result_mask = np.zeros_like(result_mask)
                        result_mask[orig_y, orig_x] = 1
                return_results = {"mask":result_mask, "max":np.max(output), "r":r}
                result_output = output
                result_new_data = new_data


    if return_ring_width > 0:
        result_mask = np.ones_like(data, dtype=np.uint8)
        r = return_results["r"]
        center = hist_data[r]["center"]
        orig_y, orig_x = disk(center, (r + return_ring_width/2)/ps/2, shape=data.shape)
        result_mask[orig_y, orig_x] = 0
        orig_y, orig_x = disk(center, (r - return_ring_width/2)/ps/2, shape=data.shape)
        result_mask[orig_y, orig_x] = 1
        return_results["mask"] = result_mask

    # print(f"Rest took {(datetime.now() - now).total_seconds()}s")
    now = datetime.now()
    if return_hist_data:
        return return_results["mask"], hist_data, return_results["r"], result_output, result_new_data
    
    
    return return_results["mask"]


















def find_grid_hole_per_file_old(current_file, to_size=100, diameter=[12000], threshold=0.005, coverage_percentage=0.5, outside_percentage=1, outside_coverage_percentage=0.05,
                             detect_ring=True, ring_width=1000, return_hist_data=False, pixel_size=None, wobble=0, high_pass=1500,crop=50, distance=0, return_ring_width=0, **kwargs):
    from datetime import datetime
    now = datetime.now()

    return_results = None
    hist_data = {}
    current_file = Path(current_file)
    if current_file.suffix != ".mrc" and pixel_size is None:
        raise TypeError(f"{current_file} is not mrc file and not pixel size was given")
    if not isinstance(diameter, (list, tuple)):
        diameter = [diameter]
    for r in diameter:
        if not detect_ring:
            y,x = disk((0,0), r/to_size / 2)
        else:
            ring_r = int((r + (ring_width // 2))/to_size / 2)

            outer_y, outer_x = disk((ring_r*2,ring_r*2), (r + (ring_width // 2))/to_size / 2)
            inner_y, inner_x = disk((ring_r*2,ring_r*2), (r - (ring_width // 2))/to_size / 2)
            ring_img = np.zeros((ring_r * 4, ring_r*4))
            ring_img[outer_y, outer_x] = 1
            ring_img[inner_y, inner_x] = 0
            y,x = np.nonzero(ring_img)
            y = np.array(y) - ring_r * 2
            x = np.array(x) - ring_r * 2
            circle_y,circle_x = disk((0,0), r/to_size / 2)
            # plt.imshow(ring_img)
            # plt.show()
            # print(len(y), len(x))
            # break

        radius = r/to_size // 2
        
    
        
        
                    
            
        data, mrc_ps = load_file(current_file)
        if pixel_size is None:
            ps = mrc_ps
        else:
            ps = pixel_size

        middle = np.median(data)
        std = np.std(data)
        left = middle - std * 4
        right = middle + std * 4
        data = np.clip(data, left, right)
        data = data -np.min(data)
        data = data / np.max(data)
        ratio = ps / to_size
        new_shape = [round(s * ratio) for s in data.shape]
        new_data = np.array(Image.fromarray(data).resize(new_shape[::-1]))

        
        if np.isclose(high_pass, 0):
            pass
            
            
        else:
            sig = int(high_pass / to_size)
            sig += (sig + 1) % 2
            new_data = gaussian_filter(new_data,0,to_size) - gaussian_filter(new_data,sig,to_size)

        if crop > 0:
            cut_off = max(1, int(crop * ratio))
            new_data = new_data[cut_off:-cut_off, cut_off:-cut_off]
        else:
            cut_off = 0
        
        
        low = int(- radius *outside_percentage)
        high_0 = int(new_data.shape[0] + radius * outside_percentage)
        high_1 = int(new_data.shape[1] + radius * outside_percentage)
        output = np.zeros((high_0 - low, high_1 - low ))
        pixels = new_data.size
        min_length = pixels * coverage_percentage
        outside_min_length = pixels * outside_coverage_percentage

        print(f"Prepare took: {(datetime.now() - now).total_seconds()}s")
        now = datetime.now()
        test_new_stuff(new_data, radius, ps)
        print(f"New mthod took: {(datetime.now() - now).total_seconds()}s")
        now = datetime.now()
        now = datetime.now()
        
        for i in range(low, high_0):
            for j in range(low, high_1):
                current_y = y + i
                current_x = x + j

                usable_idxs = np.where((current_y >= 0) & (current_y < new_data.shape[0]) & (current_x>=0) & (current_x<new_data.shape[1]))
                
                # print(new_data.shape)
                current_y = current_y[usable_idxs]
                current_x = current_x[usable_idxs]
                

                idxs = np.ravel_multi_index((current_y, current_x), new_data.shape)
                outside_values = np.delete(new_data.flatten(), idxs)
                inside_values = new_data[current_y, current_x]

                if detect_ring:
                    current_y = circle_y + i
                    current_x = circle_x + j

                    usable_idxs = np.where((current_y >= 0) & (current_y < new_data.shape[0]) & (current_x>=0) & (current_x<new_data.shape[1]))
                    
                    # print(new_data.shape)
                    current_y = current_y[usable_idxs]
                    current_x = current_x[usable_idxs]
                    

                    idxs = np.ravel_multi_index((current_y, current_x), new_data.shape)
                    length_out = len(np.delete(new_data.flatten(), idxs))
                    length_inside = len(current_y)
                else:
                    length_out = len(outside_values)
                    length_inside = len(inside_values)

                if length_out < outside_min_length:
                    output[i + int(radius * outside_percentage),j + int(radius * outside_percentage)] = 0 
                elif length_inside < min_length:
                    output[i + int(radius * outside_percentage),j + int(radius * outside_percentage)] = 0
                else:
                       
                    outside = np.nanmean(outside_values)
                    inside = np.nanmean(inside_values)
                    # outside = np.nanmedian(outside_values)
                    # inside = np.nanmedian(inside_values)
                    if detect_ring:
                        output[i + int(radius * outside_percentage),j + int(radius * outside_percentage)] = outside - inside

                    else:
                        output[i + int(radius * outside_percentage),j + int(radius * outside_percentage)] = inside - outside



        circle_center = np.unravel_index(np.argmax(output), output.shape)
        

        highest_value = np.max(output)
        wobble_steps = 0.0025
        better_result = None
        if wobble>0:
            start = (1-wobble) * r/to_size / 2
            end = (1+wobble) * r/to_size / 2
            wobble_radii = np.arange(start=start, stop=end, step=wobble_steps * r / to_size / 2, )
            wobble_radii = np.concatenate((wobble_radii, [end]))
            y_wobble = int(new_data.shape[0] * wobble)
            x_wobble = int(new_data.shape[1] * wobble)
            
            for wobble_r in wobble_radii:
                if not detect_ring:
                            
                    current_y_r,current_x_r = disk((0,0), wobble_r)

                else:
                    ring_r = int(wobble_r + (ring_width // 2)/to_size / 2)
                    # outer_y, outer_x = disk(np.array(circle_center) + low, wobble_r)
                    outer_y, outer_x = disk((ring_r*2,ring_r*2), wobble_r + (ring_width // 2)/to_size / 2)
                    inner_y, inner_x = disk((ring_r*2,ring_r*2), wobble_r - (ring_width // 2)/to_size / 2)
                    ring_img = np.zeros((ring_r * 4, ring_r*4))
                    ring_img[outer_y, outer_x] = 1
                    ring_img[inner_y, inner_x] = 0
                    y,x = np.nonzero(ring_img)
                    current_y_r = np.array(y) - ring_r * 2 
                    current_x_r = np.array(x) - ring_r * 2 
                    circle_y_r,circle_x_r = disk(((0,0)), wobble_r)
                for y_alter in range(-y_wobble, y_wobble+1):
                    for x_alter in range(-x_wobble, x_wobble+1):
                        current_circle_center = np.array(circle_center) + np.array([y_alter, x_alter])
                        current_x = current_x_r + current_circle_center[1] + low
                        current_y = current_y_r + current_circle_center[0] + low

                        circle_y = circle_y_r + current_circle_center[0] + low
                        circle_x = circle_x_r + current_circle_center[1] + low 

                        usable_idxs = np.where((current_y >= 0) & (current_y < new_data.shape[0]) & (current_x>=0) & (current_x<new_data.shape[1]))[0]
                        

                        current_y = current_y[usable_idxs]
                        current_x = current_x[usable_idxs]
                        

                        idxs = np.ravel_multi_index((current_y, current_x), new_data.shape)
                        outside_values = np.delete(new_data.flatten(), idxs)
                        inside_values = new_data[current_y, current_x]



                        if detect_ring:
                            current_y = circle_y
                            current_x = circle_x

                            usable_idxs = np.where((current_y >= 0) & (current_y < new_data.shape[0]) & (current_x>=0) & (current_x<new_data.shape[1]))
                            
                            # print(new_data.shape)
                            current_y = current_y[usable_idxs]
                            current_x = current_x[usable_idxs]
                            

                            idxs = np.ravel_multi_index((current_y, current_x), new_data.shape)
                            length_out = len(np.delete(new_data.flatten(), idxs))
                            length_inside = len(current_y)
                        else:
                            length_out = len(outside_values)
                            length_inside = len(inside_values)

                        if length_out < outside_min_length:
                            wobble_result = 0 
                        elif length_inside < min_length:
                            wobble_result = 0
                        else:
                            outside = np.nanmean(outside_values)
                            inside = np.nanmean(inside_values)
                            # outside = np.nanmedian(outside_values)
                            # inside = np.nanmedian(inside_values)
                        
                            if detect_ring:
                                wobble_result = outside - inside

                            else:
                                wobble_result = inside - outside

                       


                        if wobble_result > highest_value:
                            better_result = current_file, wobble_result, highest_value, wobble_r, radius, y_alter, x_alter, current_circle_center
                            highest_value = wobble_result 


        if better_result is not None:
            circle_center = better_result[-1]
            r = better_result[3] * to_size * 2




        # mask = np.zeros_like(output)
        # current_y = y + circle_center[0]
        # current_x = x + circle_center[1]
        # usable_idxs = np.where((current_y >= 0) & (current_y < output.shape[0]) & (current_x>=0) & (current_x<output.shape[1]))
                
                
        # current_y = current_y[usable_idxs]
        # current_x = current_x[usable_idxs]
        # mask[current_y, current_x] = 1
        

        orig_center = ( np.array(circle_center) + cut_off + low ) / (ps/to_size)
        orig_y, orig_x = disk(orig_center, r/ps / 2, shape=data.shape)
        orig_mask = np.zeros_like(data, dtype=np.uint8)
        orig_mask[orig_y, orig_x] = 1

        result_mask = np.zeros_like(data, dtype=np.uint8)
        if np.max(output) > threshold:
            result_mask[orig_y, orig_x] = 1



        values, edges = np.histogram(output, bins=50)

        hist_data[r] = {"values":values, "edges":edges, "threshold":threshold, "center":orig_center}
        if return_results is None:
            if np.abs(distance) > 0 and np.abs(distance) < r / 2:
                orig_y, orig_x = disk(orig_center, np.abs(distance)/ps, shape=data.shape)
                if distance > 0:
                    result_mask[orig_y, orig_x] = 0
                else:
                    result_mask = np.zeros_like(result_mask)
                    result_mask[orig_y, orig_x] = 1
            return_results = {"mask":result_mask, "max":np.max(output), "r":r}
            
            result_output = output
            result_new_data = new_data
        else:
            if np.max(output) > return_results["max"]:
                if np.abs(distance) > 0 and np.abs(distance) < r / 2:
                    orig_y, orig_x = disk(orig_center, np.abs(distance)/ps, shape=data.shape)
                    if distance > 0:
                        result_mask[orig_y, orig_x] = 0
                    else:
                        result_mask = np.zeros_like(result_mask)
                        result_mask[orig_y, orig_x] = 1
                return_results = {"mask":result_mask, "max":np.max(output), "r":r}
                result_output = output
                result_new_data = new_data


    if return_ring_width > 0:
        result_mask = np.ones_like(data, dtype=np.uint8)
        r = return_results["r"]
        center = hist_data[r]["center"]
        orig_y, orig_x = disk(center, (r + return_ring_width/2)/ps/2, shape=data.shape)
        result_mask[orig_y, orig_x] = 0
        orig_y, orig_x = disk(center, (r - return_ring_width/2)/ps/2, shape=data.shape)
        result_mask[orig_y, orig_x] = 1
        return_results["mask"] = result_mask

    print(f"Old method took: {(datetime.now() - now).total_seconds()}s")
    now = datetime.now()
    if return_hist_data:
        return return_results["mask"], hist_data, return_results["r"], result_output, result_new_data
    
    
    return return_results["mask"]



def find_grid_hole(to_size=100, diameter=[12000], threshold=0.005, files=[], coverage_percentage=0.5, outside_percentage=1, outside_coverage_percentage=0.05, detect_ring=True, ring_width=1000, return_hist_data=False):
    return_results = {}
    hist_data = {}
    if not isinstance(radius, (list, tuple)):
        diameter = [diameter]
    for r in diameter:
        if not detect_ring:
            y,x = disk((0,0), r/to_size / 2)
        else:
            ring_r = int((r + (ring_width // 2))/to_size / 2)

            outer_y, outer_x = disk((ring_r*2,ring_r*2), (r + (ring_width // 2))/to_size / 2)
            inner_y, inner_x = disk((ring_r*2,ring_r*2), (r - (ring_width // 2))/to_size / 2)
            ring_img = np.zeros((ring_r * 4, ring_r*4))
            ring_img[outer_y, outer_x] = 1
            ring_img[inner_y, inner_x] = 0
            y,x = np.nonzero(ring_img)
            y = np.array(y) - ring_r * 2
            x = np.array(x) - ring_r * 2
            circle_y,circle_x = disk((0,0), r/to_size / 2)
            # plt.imshow(ring_img)
            # plt.show()
            # print(len(y), len(x))
            # break

        radius = r/to_size // 2
        
        for counter, current_file in enumerate(files):
            
            current_file = Path(current_file)
            if current_file.suffix == ".mrc":            
                
                data = mrcfile.open(current_file)
                ps = data.voxel_size["x"]
                data = data.data*1
                middle = np.median(data)
                std = np.std(data)
                left = middle - std * 4
                right = middle + std * 4
                data = np.clip(data, left, right)
                data = data -np.min(data)
                data = data / np.max(data)
                ratio = ps / to_size
                new_shape = [round(s * ratio) for s in data.shape]
                new_data = np.array(Image.fromarray(data).resize(new_shape[::-1]))

                sig = int(1500 / to_size)
                sig += (sig + 1) % 2
                filtered = gaussian_filter(new_data,0,to_size) - gaussian_filter(new_data,sig,to_size)

                cut_off = 5
                new_data = filtered[cut_off:-cut_off,cut_off:-cut_off]
                low = int(- radius *outside_percentage)
                high_0 = int(new_data.shape[0] + radius * outside_percentage)
                high_1 = int(new_data.shape[1] + radius * outside_percentage)
                output = np.zeros((high_0 - low, high_1 - low ))
                pixels = filtered.size
                min_length = pixels * coverage_percentage
                outside_min_length = pixels * outside_coverage_percentage
                for i in range(low, high_0):
                    for j in range(low, high_1):
                        current_y = y + i
                        current_x = x + j

                        usable_idxs = np.where((current_y >= 0) & (current_y < new_data.shape[0]) & (current_x>=0) & (current_x<new_data.shape[1]))
                        
                        # print(new_data.shape)
                        current_y = current_y[usable_idxs]
                        current_x = current_x[usable_idxs]
                        

                        idxs = np.ravel_multi_index((current_y, current_x), new_data.shape)
                        outside_values = np.delete(new_data.flatten(), idxs)
                        inside_values = new_data[current_y, current_x]

                        if detect_ring:
                            current_y = circle_y + i
                            current_x = circle_x + j

                            usable_idxs = np.where((current_y >= 0) & (current_y < new_data.shape[0]) & (current_x>=0) & (current_x<new_data.shape[1]))
                            
                            # print(new_data.shape)
                            current_y = current_y[usable_idxs]
                            current_x = current_x[usable_idxs]
                            

                            idxs = np.ravel_multi_index((current_y, current_x), new_data.shape)
                            length_out = len(np.delete(new_data.flatten(), idxs))
                            length_inside = len(current_y)
                        else:
                            length_out = len(outside_values)
                            length_inside = len(inside_values)

                        if length_out < outside_min_length:
                            output[i + int(radius * outside_percentage),j + int(radius * outside_percentage)] = 0 
                        elif length_inside < min_length:
                            output[i + int(radius * outside_percentage),j + int(radius * outside_percentage)] = 0
                        else:
                        
                            outside = np.nanmean(outside_values)
                            inside = np.nanmean(inside_values)
                            # outside = np.nanmedian(outside_values)
                            # inside = np.nanmedian(inside_values)
                            
                            if detect_ring:
                                output[i + int(radius * outside_percentage),j + int(radius * outside_percentage)] = outside - inside

                            else:
                                output[i + int(radius * outside_percentage),j + int(radius * outside_percentage)] = inside - outside

                
                
                circle_center = np.unravel_index(np.argmax(output), output.shape)
                


                mask = np.zeros_like(output)
                current_y = y + circle_center[0]
                current_x = x + circle_center[1]
                usable_idxs = np.where((current_y >= 0) & (current_y < output.shape[0]) & (current_x>=0) & (current_x<output.shape[1]))
                        
                        
                current_y = current_y[usable_idxs]
                current_x = current_x[usable_idxs]
                mask[current_y, current_x] = 1
                


                orig_center = ( np.array(circle_center) + cut_off + low ) / (ps/to_size)
                orig_y, orig_x = disk(orig_center, r/ps / 2, shape=data.shape)
                orig_mask = np.zeros_like(data, dtype=np.uint8)
                orig_mask[orig_y, orig_x] = 1

                result_mask = np.zeros_like(data, dtype=np.uint8)
                if np.max(output) > threshold:
                    result_mask[orig_y, orig_x] = 1
                values, edges = np.histogram(output, bins=50)
                if current_file not in hist_data:
                    hist_data[current_file] = {}
                hist_data[current_file][r] = {"values":values, "edges":edges, "threshold":threshold, "center":orig_center}
                if current_file not in return_results:
                    return_results[current_file] = {"mask":result_mask, "max":np.max(output),"r":r}
                else:
                    if np.max(output) > return_results[current_file]["max"]:
                        return_results[current_file] = {"mask":result_mask, "max":np.max(output), "r":r}
    if return_hist_data:
        return {key: value["mask"] for key,value in return_results.items()}, hist_data, {key: value["r"] for key,value in return_results.items()}
    return return_results





if __name__ == "__main__":
    pass