#!/usr/bin/env python3
import os, sys
import time
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from borzoi_lora_arch_mha import BorzoiModel, EnformerModel
import grelu.resources
from torch.ao.quantization import get_default_qconfig, QConfigMapping, quantize_fx
from scipy.stats import spearmanr
from pathlib import Path
import hashlib
from torch.utils.data import DataLoader
from tqdm import tqdm
import csv
import time

MODEL_PARAMS = {
    "borzoi": {
            "input_len": 524288,
            "bin_length": 32,
            "model_class": BorzoiModel,
            "kwargs": {"n_tasks": 7611, "crop_len": 5120, "final_act_func": "softplus", "final_pool_func": None},
            "gm12878": [1288, 1345],
            "microglia": [2052],
            "smc" : [2097], 
            "spi1": [2405],
            "pai_mask": "exon", #"gene",
            "pai_metric": "SAR",
            "gtex": [7522, 7523, 7524, 7525, 7526, 7527, 7528, 7529, 7530, 7531, 7532, 7533,
        7534, 7535, 7536, 7539, 7540, 7541, 7542, 7543, 7544, 7545, 7546, 7547,
        7548, 7549, 7550, 7551, 7552, 7553, 7554, 7555, 7556, 7557, 7558, 7559,
        7560, 7561, 7562, 7563, 7564, 7565, 7566, 7567, 7568, 7569, 7570, 7571,
        7572, 7573, 7574, 7575, 7576, 7577, 7578, 7579, 7580, 7581, 7582, 7583,
        7584, 7585, 7586, 7587, 7588, 7589, 7590, 7591, 7592, 7593, 7594, 7595,
        7596, 7597, 7598, 7599, 7600, 7601, 7602, 7603, 7604, 7605, 7606, 7607,
        7608, 7609, 7610],
        "adult": [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 80, 81, 112, 113, 470, 471, 472, 473, 540, 541, 542, 543, 544, 545, 546, 547, 548, 549, 550, 551, 590, 591, 592, 593, 594, 595, 596, 597, 598, 599, 600, 601, 602, 603, 604, 605, 606, 607, 608, 609, 610, 611, 832, 833, 834, 835, 836, 837, 838, 839, 840, 841, 842, 843, 844, 845, 846, 847, 848, 849, 850, 851, 852, 853, 854, 855, 856, 857, 858, 859, 860, 861, 862, 863, 864, 865, 930, 931, 932, 933, 952, 953, 954, 955, 974, 975, 976, 977, 978, 979, 980, 981, 982, 983, 984, 985, 986, 987, 988, 989, 990, 991, 992, 993, 994, 995, 996, 997, 998, 999, 1000, 1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009, 1012, 1013, 1166, 1167, 1168, 1169, 1170, 1171, 1172, 1173, 1174, 1175, 1176, 1177, 1178, 1179, 1180, 1181, 1182, 1183, 1184, 1185, 1186, 1187, 1188, 1189, 1190, 1191, 1192, 1193, 1194, 1195, 1196, 1197, 1256, 1257, 1258, 1259, 1260, 1261, 1262, 1263, 1276, 1277, 1285, 1298, 1307, 1308, 1320, 1336, 1338, 1375, 1376, 1411, 1414, 1426, 1427, 1430, 1431, 1432, 1433, 1442, 1448, 1453, 1469, 1474, 1477, 1490, 1496, 1500, 1503, 1512, 1517, 1519, 1521, 1524, 1526, 1527, 1531, 1533, 1541, 1544, 1554, 1556, 1557, 1558, 1559, 1561, 1563, 1567, 1576, 1582, 1583, 1599, 1602, 1606, 1607, 1611, 1627, 1630, 1631, 1636, 1638, 1642, 1647, 1649, 1658, 1668, 1670, 1671, 1675, 1677, 1685, 1687, 1688, 1694, 1696, 1705, 1707, 1715, 1724, 1733, 1738, 1746, 1747, 1748, 1752, 1764, 1765, 1769, 1770, 1771, 1778, 1784, 1790, 1793, 1794, 1798, 1800, 1801, 1803, 1804, 1810, 1811, 1819, 1822, 1828, 1830, 1832, 1834, 1839, 1855, 1859, 1861, 1863, 1868, 1870, 1872, 1873, 1878, 1882, 1883, 1893, 1894, 1905, 1916, 1927, 1929, 1941, 1942, 1945, 1949, 2205, 2206, 2207, 2211, 2212, 2213, 2214, 2217, 2218, 2219, 2220, 2221, 2222, 2223, 2224, 2225, 2226, 2227, 2228, 2229, 2231, 2232, 2256, 2269, 2293, 2294, 2295, 2296, 2301, 2302, 2303, 2308, 2312, 2351, 2355, 2356, 2357, 2364, 2584, 2626, 2627, 2721, 2722, 2940, 2941, 2943, 2944, 2945, 2947, 2948, 2950, 2954, 2955, 2958, 2959, 2964, 2968, 2973, 2974, 2975, 2978, 2981, 2984, 2986, 2988, 2991, 2992, 3005, 3007, 3010, 3012, 3014, 3016, 3017, 3019, 3028, 3033, 3036, 3037, 3038, 3039, 3041, 3043, 3044, 3045, 3046, 3049, 3050, 3055, 3057, 3058, 3065, 3068, 3073, 3074, 3075, 3082, 3084, 3087, 3089, 3093, 3095, 3097, 3098, 3101, 3102, 3103, 3106, 3110, 3111, 3112, 3113, 3118, 3119, 3120, 3121, 3124, 3128, 3131, 3134, 3139, 3141, 3143, 3144, 3145, 3146, 3147, 3154, 3155, 3157, 3162, 3163, 3167, 3172, 3173, 3177, 3179, 3183, 3184, 3185, 3186, 3188, 3192, 3193, 3195, 3198, 3200, 3203, 3205, 3206, 3207, 3212, 3223, 3226, 3231, 3235, 3239, 3241, 3246, 3249, 3250, 3253, 3255, 3257, 3259, 3261, 3262, 3265, 3266, 3272, 3276, 3278, 3280, 3281, 3283, 3287, 3288, 3293, 3297, 3299, 3301, 3311, 3312, 3314, 3315, 3316, 3331, 3334, 3336, 3337, 3346, 3347, 3348, 3349, 3353, 3354, 3364, 3367, 3369, 3370, 3379, 3380, 3382, 3384, 3386, 3387, 3388, 3389, 3391, 3392, 3394, 3395, 3398, 3400, 3403, 3404, 3408, 3409, 3412, 3415, 3417, 3422, 3425, 3426, 3430, 3432, 3434, 3437, 3439, 3440, 3441, 3442, 3446, 3448, 3453, 3454, 3459, 3464, 3466, 3467, 3470, 3472, 3474, 3475, 3476, 3477, 3478, 3485, 3489, 3490, 3500, 3502, 3506, 3512, 3513, 3515, 3523, 3527, 3529, 3532, 3540, 3545, 3549, 3551, 3553, 3555, 3558, 3559, 3560, 3564, 3565, 3566, 3567, 3570, 3571, 3573, 3582, 3584, 3585, 3590, 3591, 3592, 3595, 3597, 3603, 3605, 3613, 3614, 3622, 3623, 3628, 3630, 3632, 3633, 3643, 3645, 3647, 3648, 3651, 3652, 3653, 3654, 3655, 3665, 3671, 3675, 3677, 3678, 3681, 3682, 3683, 3688, 3691, 3692, 3694, 3695, 3699, 3701, 3703, 3706, 3708, 3710, 3712, 3714, 3723, 3726, 3728, 3730, 3731, 3734, 3736, 3737, 3738, 3743, 3744, 3745, 3749, 3750, 3752, 3756, 3760, 3761, 3762, 3764, 3767, 3781, 3790, 3793, 3794, 3795, 3797, 3801, 3804, 3806, 3809, 3812, 3814, 3815, 3816, 3820, 3824, 3828, 3829, 3836, 3838, 3839, 3840, 3842, 3846, 3849, 3852, 3853, 3854, 3856, 3857, 3858, 3862, 3865, 3872, 3873, 3875, 3878, 3883, 3884, 3886, 3887, 3889, 3893, 3896, 3901, 3903, 3904, 3907, 3913, 3922, 3926, 3928, 3929, 3930, 3931, 3935, 3938, 3939, 3942, 3944, 3950, 3952, 3953, 3955, 3956, 3959, 3961, 3962, 3966, 3967, 3968, 3972, 3976, 3981, 3982, 3987, 3988, 3990, 3994, 3995, 4000, 4007, 4008, 4014, 4020, 4023, 4024, 4025, 4027, 4029, 4031, 4037, 4039, 4041, 4043, 4045, 4050, 4053, 4055, 4056, 4057, 4062, 4066, 4068, 4071, 4073, 4082, 4085, 4087, 4089, 4095, 4096, 4097, 4103, 4111, 4114, 4115, 4116, 4117, 4118, 4119, 4123, 4124, 4125, 4128, 4129, 4131, 4132, 4134, 4135, 4139, 4140, 4141, 4146, 4149, 4154, 4155, 4158, 4159, 4160, 4166, 4171, 4175, 4177, 4178, 4181, 4184, 4187, 4189, 4197, 4200, 4204, 4205, 4208, 4211, 4213, 4217, 4218, 4219, 4223, 4224, 4225, 4226, 4234, 4235, 4238, 4239, 4254, 4255, 4256, 4261, 4263, 4264, 4265, 4272, 4276, 4280, 4287, 4288, 4289, 4291, 4292, 4299, 4303, 4305, 4306, 4307, 4312, 4323, 4328, 4329, 4331, 4332, 4335, 4340, 4349, 4354, 4356, 4359, 4360, 4363, 4367, 4373, 4374, 4377, 4380, 4381, 4386, 4389, 4391, 4394, 4398, 4399, 4400, 4401, 4403, 4406, 4409, 4411, 4412, 4413, 4417, 4418, 4422, 4429, 4434, 4437, 4440, 4442, 4444, 4453, 4455, 4458, 4469, 4472, 4474, 4482, 4489, 4491, 4494, 4496, 4500, 4502, 4504, 4505, 4506, 4509, 4511, 4516, 4520, 4524, 4527, 4528, 4535, 4538, 4540, 4542, 4544, 4548, 4556, 4557, 4558, 4561, 4562, 4564, 4570, 4571, 4574, 4582, 4583, 4585, 4587, 4590, 4593, 4594, 4603, 4605, 4606, 4607, 4608, 4609, 4611, 4614, 4615, 4620, 4621, 4622, 4631, 4634, 4640, 4644, 4645, 4648, 4650, 4652, 4657, 4660, 4666, 4667, 4669, 4671, 4673, 4678, 4682, 4685, 4686, 4687, 4688, 4689, 4695, 4696, 4703, 4704, 4706, 4711, 4713, 4716, 4720, 4724, 4726, 4729, 4730, 4731, 4732, 4736, 4740, 4744, 4750, 4751, 4752, 4755, 4757, 4761, 4762, 4764, 4767, 4770, 4771, 4774, 4775, 4776, 4786, 4787, 4793, 4794, 4796, 4800, 4811, 4813, 4821, 4822, 4826, 4835, 4837, 4839, 4847, 4851, 4853, 4855, 4856, 4857, 4858, 4859, 4861, 4863, 4866, 4867, 4868, 4870, 4881, 4882, 4887, 4888, 4894, 4898, 4900, 4902, 4910, 4911, 4919, 4922, 4925, 4926, 4927, 4928, 4931, 4938, 4941, 4943, 4946, 4951, 4961, 4962, 4965, 4970, 4975, 4977, 4979, 4980, 4983, 4988, 4991, 4996, 4997, 4998, 5001, 5005, 5006, 5007, 5012, 5015, 5016, 5018, 5019, 5020, 5024, 5025, 5028, 5030, 5038, 5045, 5051, 5064, 5068, 5069, 5070, 5083, 5084, 5085, 5087, 5090, 5092, 5096, 5097, 5099, 5102, 5103, 5105, 5107, 5109, 5111, 5115, 5116, 5119, 5120, 5121, 5128, 5130, 5131, 5134, 5135, 5139, 5140, 5141, 5142, 5143, 5144, 5145, 5148, 5153, 5156, 5165, 5166, 5168, 5172, 5177, 5179, 5180, 5181, 5182, 5183, 5192, 5196, 5197, 5198, 5207, 5212, 5216, 5217, 5219, 5221, 5228, 5232, 5236, 5242, 5244, 5245, 5247, 5250, 5252, 5253, 5256, 5257, 5261, 5262, 5263, 5274, 5278, 5280, 5282, 5283, 5291, 5301, 5308, 5310, 5311, 5313, 5314, 5315, 5317, 5320, 5322, 5326, 5327, 5328, 5331, 5336, 5337, 5339, 5341, 5342, 5348, 5351, 5353, 5354, 5355, 5358, 5360, 5364, 5372, 5376, 5377, 5383, 5387, 5388, 5392, 5402, 5406, 5407, 5408, 5410, 5414, 5415, 5417, 5419, 5421, 5423, 5426, 5430, 5441, 5442, 5444, 5449, 5450, 5453, 5458, 5462, 5469, 5470, 5480, 5482, 5484, 5487, 5493, 5495, 5500, 5503, 5510, 5513, 5516, 5517, 5518, 5523, 5529, 5532, 5533, 5539, 5543, 5552, 5556, 5557, 5559, 5563, 5564, 5572, 5574, 5582, 5586, 5592, 5593, 5594, 5595, 5598, 5600, 5603, 5604, 5624, 5625, 5626, 5630, 5631, 5632, 5633, 5634, 5636, 5637, 5639, 5640, 5641, 5645, 5648, 5649, 5650, 5654, 5655, 5659, 5660, 5667, 5668, 5671, 5672, 5677, 5680, 5683, 5686, 5687, 5690, 5691, 5692, 5693, 5694, 5700, 5703, 5705, 5708, 5711, 5713, 5714, 5715, 5716, 5720, 5721, 5726, 5728, 5731, 5732, 5733, 5734, 5738, 5739, 5741, 5742, 5744, 5745, 5753, 5757, 5759, 5764, 5765, 5766, 5771, 5775, 5777, 5778, 5785, 5786, 5792, 5795, 5797, 5802, 5805, 5806, 5814, 5817, 5820, 5821, 5823, 5824, 5826, 5830, 5832, 5834, 5835, 5836, 5840, 5846, 5847, 5851, 5852, 5853, 5854, 5855, 5856, 5857, 5859, 5860, 5862, 5866, 5868, 5872, 5874, 5877, 5878, 5879, 5883, 5884, 5885, 5889, 5892, 5893, 5895, 5897, 5904, 5907, 6068, 6069, 6070, 6071, 6072, 6073, 6074, 6075, 6076, 6077, 6078, 6079, 6080, 6081, 6082, 6083, 6086, 6087, 6088, 6089, 6090, 6091, 6092, 6093, 6094, 6095, 6096, 6097, 6098, 6099, 6100, 6101, 6102, 6103, 6104, 6105, 6110, 6111, 6132, 6133, 6174, 6175, 6238, 6239, 6282, 6283, 6286, 6287, 6290, 6291, 6294, 6295, 6296, 6297, 6298, 6299, 6300, 6301, 6302, 6303, 6304, 6305, 6306, 6307, 6308, 6309, 6312, 6313, 6314, 6315, 6322, 6323, 6324, 6325, 6349, 6350, 6353, 6354, 6358, 6359, 6360, 6361, 6368, 6369, 6372, 6373, 6374, 6375, 6376, 6377, 6378, 6379, 6383, 6384, 6388, 6389, 6396, 6397, 6398, 6399, 6407, 6408, 6412, 6413, 6414, 6415, 6425, 6426, 6427, 6428, 6432, 6433, 6435, 6436, 6437, 6438, 6442, 6443, 6445, 6446, 6447, 6448, 6450, 6451, 6454, 6455, 6459, 6460, 6461, 6462, 6468, 6469, 6470, 6471, 6482, 6483, 6486, 6487, 6488, 6489, 6490, 6491, 6494, 6495, 6497, 6498, 6499, 6500, 6501, 6502, 6505, 6506, 6507, 6508, 6510, 6511, 6512, 6513, 6517, 6518, 6521, 6522, 6530, 6531, 6543, 6544, 6547, 6548, 6549, 6550, 6551, 6552, 6553, 6554, 6555, 6556, 6558, 6559, 6560, 6561, 6568, 6569, 6577, 6578, 6579, 6580, 6587, 6588, 6590, 6591, 6593, 6594, 6603, 6604, 6611, 6612, 6613, 6614, 6619, 6620, 6623, 6624, 6627, 6628, 6631, 6632, 6633, 6634, 6635, 6636, 6637, 6638, 6641, 6642, 6646, 6647, 6652, 6653, 6660, 6661, 6664, 6665, 6666, 6667, 6672, 6673, 6677, 6678, 6679, 6680, 6686, 6687, 6688, 6689, 6691, 6692, 6699, 6700, 6706, 6707, 6711, 6712, 6716, 6717, 6721, 6722, 6726, 6727, 6730, 6731, 6732, 6739, 6740, 6741, 6742, 6744, 6745, 6747, 6748, 6749, 6750, 6753, 6754, 6768, 6769, 6770, 6771, 6772, 6773, 6774, 6775, 6776, 6780, 6781, 6782, 6783, 6784, 6785, 6787, 6788, 6789, 6790, 6794, 6795, 6799, 6800, 6812, 6813, 6816, 6817, 6818, 6819, 6822, 6823, 6824, 6825, 6826, 6827, 6829, 6830, 6832, 6833, 6834, 6835, 6842, 6843, 6846, 6847, 6850, 6851, 6852, 6853, 6854, 6855, 6856, 6862, 6863, 6864, 6865, 6866, 6867, 6868, 6879, 6880, 6883, 6884, 6885, 6886, 6889, 6890, 6892, 6893, 6896, 6897, 6902, 6903, 6906, 6907, 6910, 6911, 6912, 6913, 6914, 6915, 6916, 6917, 6920, 6921, 6922, 6923, 6927, 6928, 6931, 6932, 6935, 6936, 6937, 6938, 6942, 6943, 6944, 6945, 6954, 6955, 6960, 6961, 6962, 6966, 6967, 6970, 6971, 6987, 6988, 6989, 6990, 6996, 6997, 6999, 7000, 7004, 7005, 7008, 7009, 7017, 7018, 7022, 7023, 7027, 7028, 7035, 7036, 7037, 7038, 7040, 7041, 7042, 7043, 7048, 7049, 7054, 7055, 7056, 7057, 7060, 7061, 7062, 7063, 7064, 7065, 7066, 7067, 7068, 7069, 7072, 7073, 7082, 7083, 7084, 7085, 7088, 7089, 7093, 7094, 7097, 7098, 7111, 7112, 7113, 7114, 7122, 7123, 7124, 7125, 7128, 7129, 7130, 7131, 7132, 7133, 7135, 7136, 7137, 7138, 7141, 7142, 7145, 7146, 7148, 7149, 7150, 7151, 7165, 7166, 7173, 7174, 7187, 7188, 7189, 7190, 7191, 7192, 7193, 7194, 7195, 7196, 7198, 7199, 7211, 7212, 7215, 7216, 7220, 7221, 7222, 7223, 7224, 7225, 7229, 7230, 7237, 7238, 7239, 7240, 7241, 7244, 7245, 7246, 7247, 7254, 7255, 7267, 7268, 7269, 7270, 7275, 7276, 7277, 7278, 7279, 7280, 7285, 7286, 7291, 7292, 7293, 7294, 7295, 7296, 7300, 7301, 7302, 7303, 7304, 7305, 7311, 7312, 7313, 7314, 7315, 7316, 7319, 7320, 7321, 7322, 7323, 7324, 7325, 7326, 7327, 7328, 7329, 7332, 7333, 7336, 7337, 7343, 7344, 7345, 7346, 7347, 7348, 7349, 7350, 7352, 7353, 7354, 7355, 7361, 7362, 7363, 7367, 7368, 7370, 7371, 7373, 7374, 7377, 7378, 7379, 7380, 7396, 7397, 7407, 7408, 7413, 7414, 7415, 7416, 7419, 7420, 7423, 7428, 7429, 7430, 7431, 7432, 7433, 7434, 7435, 7436, 7437, 7444, 7445, 7446, 7447, 7459, 7460, 7461, 7463, 7464, 7467, 7468, 7469, 7470, 7472, 7473, 7478, 7479, 7494, 7495, 7499, 7500, 7502, 7503, 7505, 7506, 7509, 7510, 7511, 7512, 7514, 7515, 7516, 7517, 7520, 7521]




    },
    "enformer": {
            "input_len": 196608,
            "bin_length": 128,
            "model_class": EnformerModel,
            "kwargs": {"n_tasks": 5313, "crop_len": 320, "final_act_func": "softplus", "final_pool_func": None},
            "gm12878": [12, 69],
            "microglia": [41, 131, 392, 508, 517],
            "smc" : [82], 
            "spi1": [907],
            "pai_mask": "all",
            "pai_metric": "SAD",
    }

    }


def initialize_models(k_l: int, quant: bool, model_name = "borzoi", full = False):

    if quant == True:
        dev = "cpu"
    else:
        dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if full != True:

        model = MODEL_PARAMS[model_name]["model_class"](k_l =k_l, device = dev, **MODEL_PARAMS[model_name]["kwargs"])
        # model = BorzoiModel(k_l =k_l, k_c = k_c, n_tasks = 7611, crop_len=5120, final_act_func="softplus", final_pool_func=None)
        state_dict = torch.load(f'{model_name}_lora_weights/{model_name}_lora_lr{k_l}.pth', weights_only=True)

        model.load_state_dict(state_dict, strict = True)
    else:
        if model_name == "borzoi":
            model = torch.load("./full_models/borzoi_human_rep0.pt")
        elif model_name == "enformer":
            model = torch.load("./full_models/enformer_human.pt")
        else:
            model = None

  
    return model

def one_hot_encode_dna(seqs):
    """One-hot encode a list of DNA sequences (A,C,G,T)."""
    base_to_idx = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    N = len(seqs)
    L = len(seqs[0])
    
    # Create tensor of zeros
    one_hot = torch.zeros((N, 4, L), dtype=torch.float32)
    
    # Fill positions
    for i, seq in enumerate(seqs):
        for j, base in enumerate(seq):
            if base in base_to_idx:  # ignore Ns or ambiguous bases
                one_hot[i, base_to_idx[base], j] = 1.0
    return one_hot

class SeqDataset(torch.utils.data.Dataset):
    def __init__(self, input_len, num_seqs):
        self.input_len = input_len
        self.num_seqs = num_seqs
        self.bases = 4
    
    def __len__(self):
        return self.num_seqs
    def __getitem__(self, idx):

        rand_indices = np.random.randint(0, self.bases, size=(1, self.input_len))
        one_hot_seqs = np.zeros((1, self.bases, self.input_len), dtype=np.float32)
        one_hot_seqs[0, rand_indices, np.arange(self.input_len)] = 1.0
        embedding = torch.from_numpy(one_hot_seqs).to('cpu')
        return embedding



def quantize_and_save(model: torch.nn.Module, model_name: str, outfile: str, device="cpu", rank = None):
    """Quantize a LoRA Borzoi model and save weights + metrics."""
    seq_count = 16
    device = "cpu"
    seq_len = MODEL_PARAMS[model_name]["input_len"]

    dataset = SeqDataset(input_len=seq_len, num_seqs=seq_count)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    print("loaded data")
    model = model.to(device)
    # model = model.model
    model.eval()

    sample_batch = next(iter(loader))
    sample_batch = sample_batch[0].to(device)
    _ = model(sample_batch)
    # --- FX Quantization ---
    qconfig = get_default_qconfig("fbgemm")
    qconfig_mapping = QConfigMapping().set_global(qconfig)
    prepared_model = quantize_fx.prepare_fx(model, qconfig_mapping, sample_batch)
    print("prepping model")
#    with torch.no_grad():
#        for batch in tqdm(loader, desc="Prepping quant model"):
#            prepared_model(batch[0])
    model_quantized = quantize_fx.convert_fx(prepared_model)

    print("Evaluating quantized model ...")
    # --- Evaluate metrics ---
    dataset_2 = SeqDataset(input_len=seq_len, num_seqs=seq_count)
    loader_2 = DataLoader(dataset_2, batch_size=1, shuffle=False, num_workers=0)
    rmaes, pcors, scors, maes = [], [], [], []
    with torch.no_grad():
        for batch in loader_2:
            y = model(batch[0]).flatten()
            yhat = model_quantized(batch[0]).flatten()

            corr = torch.corrcoef(torch.stack([y, yhat]))
            pcors.append(float(corr[0,1]))

            scorr, _ = spearmanr(y.cpu().numpy(), yhat.cpu().numpy())
            scors.append(float(scorr))

            mae = torch.mean(torch.abs(yhat - y))
            maes.append(float(mae))
            rmaes.append(float(mae / torch.mean(torch.abs(y))))

    # Print metrics
    print(f"Quantization Metrics for {outfile}:")
    print(f"  Relative MAE: {np.nanmean(rmaes):.4f} ± {np.std(rmaes):.4f}")
    print(f"  MAE: {np.nanmean(maes):.4f} ± {np.std(maes):.4f}")
    print(f"  Pearson: {np.nanmean(pcors):.4f} ± {np.std(pcors):.4f}")
    print(f"  Spearman: {np.nanmean(scors):.4f} ± {np.std(scors):.4f}")

    metrics = {
        "rank": rank,
        "rmae_mean": round(np.nanmean(rmaes), 4),
        "rmae_std": round(np.std(rmaes), 4),
        "mae_mean": round(np.nanmean(maes), 4),
        "mae_std": round(np.std(maes), 4),
        "pearson_mean": round(np.nanmean(pcors), 4),
        "pearson_std": round(np.std(pcors), 4),
        "spearman_mean": round(np.nanmean(scors), 4),
        "spearman_std": round(np.std(scors), 4),
    }

    file_exists = os.path.isfile(f"{model_name}_quantization_metrics.tsv")
    with open(f"{model_name}_quantization_metrics.tsv", "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=metrics.keys(), delimiter="\t")
            if not file_exists:
                writer.writeheader()
            writer.writerow(metrics)

            print(f"Results saved to {model_name}_quantization_metrics.tsv")

    torch.save(model_quantized.state_dict(), outfile)
    print(f"Saved quantized weights: {outfile}")
    return model_quantized

def get_linear_params(model: torch.nn.Module):
    """Return total number of parameters in Linear layers recursively."""
    total = 0
    for name, module in model.named_children():
        if isinstance(module, torch.nn.Linear):
            total += sum(p.numel() for p in module.parameters())
        else:
            total += get_linear_params(module)
    return total

def get_model_size(model: torch.nn.Module):
    """Return total number of parameters in the model."""
    return sum(p.numel() for p in model.parameters())

def save_model_sizes(quant, name, outpath="benchmark_model_sizes.tsv", rank_index=None):
    """Generate TSV of model sizes for all ranks and configs."""
    all_ranks = [1,2,4,8,16,32,64,128,256,512, "full"]
    if rank_index is None or not (1 <= rank_index <= len(all_ranks)):
        raise ValueError(f"rank_index must be between 1 and {len(all_ranks)}")

    rank = all_ranks[rank_index - 1]
    if quant == True:
        device = "cpu"
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    results = []
    print(f"\n[INFO] Starting model: {name}, quant={quant}, rank={rank}")

    model = initialize_models(k_l=rank, quant=quant, model_name=name)

    if quant:
        outfile = f"{name}_lora_weights/{name}_lora_lr{rank}_quant.pth"
        print(f"[INFO] Quantizing {name} model rank {rank} ...")
        model = quantize_and_save(model, model_name=name, outfile=outfile, device=device, rank =rank)
        print("[INFO] Quantization complete.")
    else:
        n_params = get_model_size(model)
        n_linear_params = get_linear_params(model)
        results.append({
            "model": name,
            "rank": rank,
            "quant": quant,
            "n_params": n_params,
            "n_linear_params": n_linear_params
        })
        print(f"[INFO] Rank={rank}, quant={quant} -> n_params={n_params}")

        # Append or create results file
        df = pd.DataFrame(results)
        if os.path.exists(outpath):
            df.to_csv(outpath, sep="\t", mode="a", index=False, header=False)
        else:
            df.to_csv(outpath, sep="\t", index=False)
        print(f"[INFO] Results appended to {outpath}")
        return df


def main():
    """
    Usage:
      python grelu_quantize.py <rank_index> [model_name] [quant]
    Example:
      python grelu_quantize.py 3 borzoi False
      python grelu_quantize.py 5 enformer True
    """
    if len(sys.argv) < 2:
        print("Usage: python grelu_quantize.py <rank_index> [model_name] [quant]")
        sys.exit(1)

    try:
        rank_index = int(sys.argv[1])
    except ValueError:
        print("Error: rank_index must be an integer from 1 to 11.")
        sys.exit(1)

    model_name = sys.argv[2] if len(sys.argv) > 2 else "enformer"
    quant = sys.argv[3].lower() == "true" if len(sys.argv) > 3 else True

    outpath = f"benchmark_{model_name}_model_sizes.tsv"

    print(f"\n[RUNNING] rank_index={rank_index}, model={model_name}, quant={quant}")
    save_model_sizes(quant=quant, name=model_name, outpath=outpath, rank_index=rank_index)


if __name__ == "__main__":
    main()


