from data_processing.sample_transforms import *
from datasets.synthetic_dataset_utils import *
from datasets.luces_dataset import ScaleByPhi, NormalizeByMeanDepthLuces
from data_processing.pad_to_power2 import *
from data_processing.crop_based_on_mask import *

#synthetic datasets
syn_training_transforms = transforms.Compose([
      ErodeMask(keys=['mask'],mask_erosion_size=(6,6)),
      RandomCropResize((512,512),keys=['mask','depth','normal']),
      NormalizeByMeanDepthSynthetic(),
      MyToTensor(keys=['mask','depth','normal'])
      ])
      
syn_test_transforms = transforms.Compose([
      ErodeMask(keys=['mask'],mask_erosion_size=(6,6)),
      RandomCropResize((512,512),keys=['mask','depth','normal']),
      NormalizeByMeanDepthSynthetic(),
      MyToTensor(keys=['mask','depth','normal'])
      ])

sculpture_training_transforms = transforms.Compose([
    # ErodeMask(keys=['mask'], mask_erosion_size=(2, 2)),
    # RandomCropResize((512, 512), keys=['mask', 'depth', 'normal']),
    NormalizeByMeanDepthSynthetic(),
    MyToTensor(keys=['mask', 'depth', 'normal'])
])

sculpture_test_transforms = transforms.Compose([
    # ErodeMask(keys=['mask'], mask_erosion_size=(2, 2)),
    NormalizeByMeanDepthSynthetic(),
    MyToTensor(keys=['mask', 'depth', 'normal'])
])

#luces data loading
luces_transforms = transforms.Compose([
      #Resize((682,512),keys=['normal','depth','mask'],list_keys=['images']),
       # ScaleByPhi(),
       NormalizeByMeanDepthLuces(),
       PadSquareToPower2Intrinsics(keys=['normal','depth','mask'],list_keys=[]),
       MyToTensor(keys=['normal','depth','mask'],list_keys=[])
     ])

diligent_transforms = transforms.Compose([
      #Resize((682,512),keys=['normal','depth','mask'],list_keys=['images']),
       # ScaleByPhi(),
       NormalizeByMeanDepthSynthetic(),
       PadSquareToPower2Intrinsics(keys=['normal','depth','mask'],list_keys=[]),
       MyToTensor(keys=['normal','depth','mask'],list_keys=[])
     ])

real_transforms = transforms.Compose([
      #Resize((682,512),keys=['normal','depth','mask'],list_keys=['images']),
       # ScaleByPhi(),
       # NormalizeByMeanDepthSynthetic(),
       PadSquareToPower2Intrinsics(keys=['normal','mask'],list_keys=[]),
       MyToTensor(keys=['normal','mask'],list_keys=[])
     ])
     

