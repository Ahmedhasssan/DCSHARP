import cv2
import os

# Set the directory containing the images
image_folder = '/home/ah2288/gs_baseline/gaussian-splatting/output_new/playroom/test/ours_30000/renders'
video_name = 'playroom_video_fps3.avi'

# Get all image filenames from the directory and sort them if necessary
images = [img for img in os.listdir(image_folder) if img.endswith((".png", ".jpg", ".jpeg"))]
images.sort()

# Read the first image to get frame dimensions
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

# Define the video codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # 'XVID' is the codec for AVI files
video = cv2.VideoWriter(video_name, fourcc, 3, (width, height))  # 1 frame per second

# Loop over the images and add each one to the video
for image in images:
    img_path = os.path.join(image_folder, image)
    img = cv2.imread(img_path)
    video.write(img)

# Release the video writer object
video.release()
# cv2.destroyAllWindows()