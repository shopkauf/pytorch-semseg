from scipy import misc
from scipy import ndimage
from skimage.filters import gaussian
from skimage import io
import os

# input_folder = r'C:\data\Synthetic_blur_MPI_data\original'
# output_folder = r'C:\data\Synthetic_blur_MPI_data\blur_kernel_10'
# for root, directories, filenames in os.walk(input_folder):
    # for filename in filenames:
        # out_ffname = os.path.join(output_folder, 'kern010_'+filename)
        # if not os.path.exists(out_ffname):
            # im = misc.imread(os.path.join(root, filename)) 
            # blurred = gaussian(im, sigma=10,multichannel=True)  
            # misc.imsave(out_ffname, blurred)


### rename files
# input_folder = r'C:\data\Synthetic_blur_MPI_data\blur_kernel_10'
# for root, directories, filenames in os.walk(input_folder):
    # for filename in filenames:
        # out_ffname = os.path.join(input_folder, filename)
        # if os.path.exists(out_ffname):
            # new_ffname = os.path.join(input_folder, 'kern010_'+filename)
            # os.rename(out_ffname, new_ffname)

                

## ground truth mask for original images
# input_folder = r'C:\data\Synthetic_blur_MPI_data\original'
# output_folder = r'C:\data\Synthetic_blur_MPI_data\images\training'
# for root, directories, filenames in os.walk(input_folder):
    # for filename in filenames:
        # in_ffname = os.path.join(input_folder, filename)
        # out_ffname = os.path.join(output_folder, filename[:-4]+'_seg.png')
        # if os.path.exists(in_ffname):
            # im = misc.imread(os.path.join(root, filename)) 
            # mask = im * 0
            # misc.imsave(out_ffname, mask)


## ground truth mask for blurred images
# input_folder = r'C:\data\Synthetic_blur_MPI_data\blur_kernel_10'
# output_folder = r'C:\data\Synthetic_blur_MPI_data\images\training'
# for root, directories, filenames in os.walk(input_folder):
    # for filename in filenames:
        # in_ffname = os.path.join(input_folder, filename)
        # out_ffname = os.path.join(output_folder, filename[:-4]+'_seg.png')
        # if os.path.exists(in_ffname):
            # im = misc.imread(os.path.join(root, filename)) 
            # mask = im * 0
            # mask = mask + 1
            # misc.imsave(out_ffname, mask)



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
            cropped = im[2500:2800,2500:2800,0:3]
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
            cropped = im[3100:3400,2200:2500,0:3]
            misc.imsave(out_ffname, cropped)
            
            out_ffname_2 = os.path.join(output_folder_2, filename[:-4]+'_seg.png')
            mask = cropped * 0
            mask = mask[:,:,0]
            mask = mask + 1		
            misc.imsave(out_ffname_2, mask)
            
			
python train.py --dataset mpiblur --arch unet --img_rows 300 --img_cols 300

