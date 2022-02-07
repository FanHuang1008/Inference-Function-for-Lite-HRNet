# Demo-Function-for-Lite-HRNet
For the requirements, installation, training, and testing of Lite-HRNet, please visit the official GitHub page (https://github.com/HRNet/Lite-HRNet). If you fail to install mmcv-full, install mmcv and mmpose instead. 


Unlike other HRNet family, Lite-HRNet uses dataloader to load all images into the model. I’ve tried to create my own dataloader but the result was incorrect. 

![image](https://user-images.githubusercontent.com/57203983/152774358-fcd7c473-77e9-4089-bc25-d7c41f875029.png)


After reading the configuration file thoroughly, I realized that all images will be processed by val_pipeline during validation.

![image](https://user-images.githubusercontent.com/57203983/152774573-2dc25b64-f93e-44ce-9e05-566cfba89303.png)


Therefore, I used OpenCV to load a single image and imitated the whole process of val_pipeline. 

![image](https://user-images.githubusercontent.com/57203983/152774701-9261cf29-7804-4b36-87b4-c041c1263cc6.png)


The input of the model needs to be a dictionary, so I created an empty one and used img_trans as the value for ‘img.’ As for ‘img_metas’, it is originally used to draw bounding box on the image from the json file. Since we don’t need to know the ground-truth when demoing, ‘image_file’, ‘bbox-score’, and ‘bbox_id’ are not important. I assume ‘rotation’ will rotate the bounding box, so I just use the default value. Hence, center and scale are the only two parameters that will affect the demo result.

![image](https://user-images.githubusercontent.com/57203983/152775714-4ec53137-4cc4-4e7a-a378-765bb450b384.png)


Since I was too lazy to write a function that can draw the joints and limbs of a person, I borrowed the add_joints function from Efficient HRNet (https://github.com/TeCSAR-UNCC/EfficientHRNet/blob/main/lib/utils/vis.py). 

![image](https://user-images.githubusercontent.com/57203983/152775800-11599e96-ea13-41a7-aeb8-c4ca5623316e.png)
![image](https://user-images.githubusercontent.com/57203983/152775811-417565d4-35a7-425f-a550-45c97ccb8cc8.png)
