# Inference Function for Lite-HRNet
For the requirements, installation, training, and testing of Lite-HRNet, please visit the official GitHub page (https://github.com/HRNet/Lite-HRNet). If fail to install mmcv-full, install mmcv and mmpose instead. 


Unlike other HRNet family, Lite-HRNet uses dataloader to load all images into the model. I’ve tried to create my own dataloader but the result was incorrect. 

```
# build the dataloader
dataset = build_dataset(cfg.data.val, dict(test_mode=True))
dataloader_setting = dict(
    samples_per_gpu=1,
    workers_per_gpu=cfg.data.get('workers_per_gpu', 1),
    dist=distributed,
    shuffle=False,
    drop_last=False
)
dataloader_setting = dict(dataloader_setting,
                          **cfg.data.get('val_dataloader', {}))
data_loader = build_dataloader(dataset, **dataloader_setting)
```

After reading the configuration file thoroughly, I realized that all images will be processed by val_pipeline during validation.

```
val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='TopDownAffine'),
    dict(type='ToTensor'),
    dict(
        type='NormalizeTensor',
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
    dict(
        type='Collect',
        keys=[
            'img',
        ],
        meta_keys=[
            'image_file', 'center', 'scale', 'rotation', 'bbox_score',
            'flip_pairs'
        ]
    ),
]
```

Therefore, I used OpenCV to load a single image and imitated the whole process of val_pipeline.

```
# Load image from file
img = cv2.imread('test.jpg')

# Affine transform the image to make input
width = 384
height = 288
c = np.array([432, 364])
s = np.array([0.92, 1.22])
r = 0
image_size = [height, width]
trans = get_affine_transform(c, s, r, image_size)
img_trans = cv2.warpAffine(img, trans, (int(image_size[0]), int(image_size[1])), flags=cv2.INTER_LINEAR)

# Transform image to Tensor
img_trans = F.to_tensor(img_trans)

# Normalize the Tensor with mean and std
mean = [0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]
img_trans = F.normalize(img_trans, mean=mean, std=std)
img_trans = torch.reshape(img_trans, (1, 3, width, height))
```

The input of the model needs to be a dictionary, so I created an empty one and used img_trans as the value for ‘img.’ As for ‘img_metas’, it is originally used to draw bounding box on the image from the json file. Since we don’t need to know the ground-truth when inferencing, ‘image_file’, ‘bbox-score’, and ‘bbox_id’ are not important. I assumed ‘rotation’ will rotate the bounding box, so I just used the default value. Hence, center and scale are the only two parameters that will affect the inference result.

```
data = {}
data['img'] = img_trans
data['img_metas'] = [{'image_file': '', 'center': c, 'scale': s, 'rotation': r,
                      'bbox_score': 1, 'flip_pairs': [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16]], 
                      'bbox_id': 0}]
```

Since I was too lazy to write a function that can draw the joints and limbs of a person, I borrowed the add_joints function from Efficient HRNet (https://github.com/TeCSAR-UNCC/EfficientHRNet/blob/main/lib/utils/vis.py).

```
def add_joints(image, joints):
    coco_part_labels = [
    'nose', 'eye_l', 'eye_r', 'ear_l', 'ear_r',
    'sho_l', 'sho_r', 'elb_l', 'elb_r', 'wri_l', 'wri_r',
    'hip_l', 'hip_r', 'kne_l', 'kne_r', 'ank_l', 'ank_r']

    coco_part_idx = {
    b: a for a, b in enumerate(coco_part_labels)}

    coco_part_orders = [
    ('nose', 'eye_l'), ('eye_l', 'eye_r'), ('eye_r', 'nose'),
    ('eye_l', 'ear_l'), ('eye_r', 'ear_r'), ('ear_l', 'sho_l'),
    ('ear_r', 'sho_r'), ('sho_l', 'sho_r'), ('sho_l', 'hip_l'),
    ('sho_r', 'hip_r'), ('hip_l', 'hip_r'), ('sho_l', 'elb_l'),
    ('elb_l', 'wri_l'), ('sho_r', 'elb_r'), ('elb_r', 'wri_r'),
    ('hip_l', 'kne_l'), ('kne_l', 'ank_l'), ('hip_r', 'kne_r'),
    ('kne_r', 'ank_r')]

    part_idx = coco_part_idx
    part_orders = coco_part_orders

    def link(a, b):
        if part_idx[a] < joints.shape[0] and part_idx[b] < joints.shape[0]:
            jointa = joints[part_idx[a]]
            jointb = joints[part_idx[b]]
 
            if jointa[2] > 0.2 and jointb[2] > 0.2:
                cv2.line(image, (int(jointa[0]), int(jointa[1])), (int(jointb[0]), int(jointb[1])), (255, 255, 255), 3)

    # add joints
    for joint in joints:
        if joint[2] > 0.2:
            cv2.circle(image, (int(joint[0]), int(joint[1])), 3, (0, 255, 255), 7)

    # add link
    for pair in part_orders:
        link(pair[0], pair[1])

    return image
```