from scipy import misc
from scipy import ndimage
from skimage.filters import gaussian
from skimage import io
import os


### crop
from skimage import io
from scipy import misc
import os
input_folder = r'C:\data\Synthetic_blur_MPI_data\original\sharp'
output_folder = r'C:\data\Synthetic_blur_MPI_data\images\training'
output_folder_2 = r'C:\data\Synthetic_blur_MPI_data\images\labels'
for root, directories, filenames in os.walk(input_folder):
    for filename in filenames:
        in_ffname = os.path.join(input_folder, filename)
        out_ffname = os.path.join(output_folder, filename[:-4]+'.png')
        if os.path.exists(in_ffname):
            im = io.imread( in_ffname ) 
            cropped = im[2000:3024,2000:3024,0:3]
            misc.imsave(out_ffname, cropped)
            
            out_ffname_2 = os.path.join(output_folder_2, filename[:-4]+'_seg.png')
            mask = cropped * 0
            mask = mask[:,:,0]
            misc.imsave(out_ffname_2, mask)

			
from skimage import io
from scipy import misc
import os
input_folder = r'C:\data\Synthetic_blur_MPI_data\original\blur_kernel_10'
output_folder = r'C:\data\Synthetic_blur_MPI_data\images\training'
output_folder_2 = r'C:\data\Synthetic_blur_MPI_data\images\labels'
for root, directories, filenames in os.walk(input_folder):
    for filename in filenames:
        in_ffname = os.path.join(input_folder, filename)
        out_ffname = os.path.join(output_folder, filename[:-4]+'.png')
        if os.path.exists(in_ffname):
            im = io.imread( in_ffname ) 
            cropped = im[1000:2024,1000:2024,0:3]
            misc.imsave(out_ffname, cropped)
            
            out_ffname_2 = os.path.join(output_folder_2, filename[:-4]+'_seg.png')
            mask = cropped * 0
            mask = mask[:,:,0]
            mask = mask + 1		
            misc.imsave(out_ffname_2, mask)
            
			
python train.py --dataset mpiblur --arch unet --img_rows 300 --img_cols 300

