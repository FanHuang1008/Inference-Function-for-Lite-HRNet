# Inference-Function-for-Lite-HRNet
For the requirements, installation, training, and testing of Lite-HRNet, please visit the official GitHub page (https://github.com/HRNet/Lite-HRNet). If you fail to install mmcv-full, install mmcv and mmpose instead. 


Unlike other HRNet family, Lite-HRNet uses dataloader to load all images into the model. I’ve tried to create my own dataloader but the result was incorrect. 

![image](https://user-images.githubusercontent.com/57203983/152776219-6b8f9ab3-30c2-4eca-a73d-5f81f229cd4b.png)


After reading the configuration file thoroughly, I realized that all images will be processed by val_pipeline during validation.

![image](https://user-images.githubusercontent.com/57203983/152776245-6ac865e9-9e2c-43a5-baca-f46163fafb70.png)


Therefore, I used OpenCV to load a single image and imitated the whole process of val_pipeline. 

![image](https://user-images.githubusercontent.com/57203983/152776285-a4729fc6-2510-4c06-bc86-82dc514d786a.png)


The input of the model needs to be a dictionary, so I created an empty one and used img_trans as the value for ‘img.’ As for ‘img_metas’, it is originally used to draw bounding box on the image from the json file. Since we don’t need to know the ground-truth when inferencing, ‘image_file’, ‘bbox-score’, and ‘bbox_id’ are not important. I assumed ‘rotation’ will rotate the bounding box, so I just used the default value. Hence, center and scale are the only two parameters that will affect the inference result.

![image](https://user-images.githubusercontent.com/57203983/152776314-8d562952-19f0-43d3-8e82-39a246b9d5b7.png)


Since I was too lazy to write a function that can draw the joints and limbs of a person, I borrowed the add_joints function from Efficient HRNet (https://github.com/TeCSAR-UNCC/EfficientHRNet/blob/main/lib/utils/vis.py). 

![image](https://user-images.githubusercontent.com/57203983/152776349-a7efc2c6-681c-47db-8a9b-ebe3b9443f72.png)
![image](https://user-images.githubusercontent.com/57203983/152776366-b1b0335b-19f1-4845-bf8f-4a43e666ce47.png)
