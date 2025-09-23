import torch
import numpy as np
import random
from sklearn.metrics import classification_report, f1_score, confusion_matrix, accuracy_score
import torch.nn as nn
import torch.nn.functional as F


c_names = {
    'aliceblue':            '#F0F8FF',
    'antiquewhite':         '#FAEBD7',
    'aqua':                 '#00FFFF',
    'aquamarine':           '#7FFFD4',
    'azure':                '#F0FFFF',
    'beige':                '#F5F5DC',
    'bisque':               '#FFE4C4',
    'black':                '#000000',
    'blanchedalmond':       '#FFEBCD',
    'blue':                 '#0000FF',
    'blueviolet':           '#8A2BE2',
    'brown':                '#A52A2A',
    'burlywood':            '#DEB887',
    'cadetblue':            '#5F9EA0',
    'chartreuse':           '#7FFF00',
    'chocolate':            '#D2691E',
    'coral':                '#FF7F50',
    'cornflowerblue':       '#6495ED',
    'cornsilk':             '#FFF8DC',
    'crimson':              '#DC143C',
    'cyan':                 '#00FFFF',
    'darkblue':             '#00008B',
    'darkcyan':             '#008B8B',
    'darkgoldenrod':        '#B8860B',
    'darkgray':             '#A9A9A9',
    'darkgreen':            '#006400',
    'darkkhaki':            '#BDB76B',
    'darkmagenta':          '#8B008B',
    'darkolivegreen':       '#556B2F',
    'darkorange':           '#FF8C00',
    'darkorchid':           '#9932CC',
    'darkred':              '#8B0000',
    'darksalmon':           '#E9967A',
    'darkseagreen':         '#8FBC8F',
    'darkslateblue':        '#483D8B',
    'darkslategray':        '#2F4F4F',
    'darkturquoise':        '#00CED1',
    'darkviolet':           '#9400D3',
    'deeppink':             '#FF1493',
    'deepskyblue':          '#00BFFF',
    'dimgray':              '#696969',
    'dodgerblue':           '#1E90FF',
    'firebrick':            '#B22222',
    'floralwhite':          '#FFFAF0',
    'forestgreen':          '#228B22',
    'fuchsia':              '#FF00FF',
    'gainsboro':            '#DCDCDC',
    'ghostwhite':           '#F8F8FF',
    'gold':                 '#FFD700',
    'goldenrod':            '#DAA520',
    'gray':                 '#808080',
    'green':                '#008000',
    'greenyellow':          '#ADFF2F',
    'honeydew':             '#F0FFF0',
    'hotpink':              '#FF69B4',
    'indianred':            '#CD5C5C',
    'indigo':               '#4B0082',
    'ivory':                '#FFFFF0',
    'khaki':                '#F0E68C',
    'lavender':             '#E6E6FA',
    'lavenderblush':        '#FFF0F5',
    'lawngreen':            '#7CFC00',
    'lemonchiffon':         '#FFFACD',
    'lightblue':            '#ADD8E6',
    'lightcoral':           '#F08080',
    'lightcyan':            '#E0FFFF',
    'lightgoldenrodyellow': '#FAFAD2',
    'lightgreen':           '#90EE90',
    'lightgray':            '#D3D3D3',
    'lightpink':            '#FFB6C1',
    'lightsalmon':          '#FFA07A',
    'lightseagreen':        '#20B2AA',
    'lightskyblue':         '#87CEFA',
    'lightslategray':       '#778899',
    'lightsteelblue':       '#B0C4DE',
    'lightyellow':          '#FFFFE0',
    'lime':                 '#00FF00',
    'limegreen':            '#32CD32',
    'linen':                '#FAF0E6',
    'magenta':              '#FF00FF',
    'maroon':               '#800000',
    'mediumaquamarine':     '#66CDAA',
    'mediumblue':           '#0000CD',
    'mediumorchid':         '#BA55D3',
    'mediumpurple':         '#9370DB',
    'mediumseagreen':       '#3CB371',
    'mediumslateblue':      '#7B68EE',
    'mediumspringgreen':    '#00FA9A',
    'mediumturquoise':      '#48D1CC',
    'mediumvioletred':      '#C71585',
    'midnightblue':         '#191970',
    'mintcream':            '#F5FFFA',
    'mistyrose':            '#FFE4E1',
    'moccasin':             '#FFE4B5',
    'navajowhite':          '#FFDEAD',
    'navy':                 '#000080',
    'oldlace':              '#FDF5E6',
    'olive':                '#808000',
    'olivedrab':            '#6B8E23',
    'orange':               '#FFA500',
    'orangered':            '#FF4500',
    'orchid':               '#DA70D6',
    'palegoldenrod':        '#EEE8AA',
    'palegreen':            '#98FB98',
    'paleturquoise':        '#AFEEEE',
    'palevioletred':        '#DB7093',
    'papayawhip':           '#FFEFD5',
    'peachpuff':            '#FFDAB9',
    'peru':                 '#CD853F',
    'pink':                 '#FFC0CB',
    'plum':                 '#DDA0DD',
    'powderblue':           '#B0E0E6',
    'purple':               '#800080',
    'red':                  '#FF0000',
    'rosybrown':            '#BC8F8F',
    'royalblue':            '#4169E1',
    'saddlebrown':          '#8B4513',
    'salmon':               '#FA8072',
    'sandybrown':           '#FAA460',
    'seagreen':             '#2E8B57',
    'seashell':             '#FFF5EE',
    'sienna':               '#A0522D',
    'silver':               '#C0C0C0',
    'skyblue':              '#87CEEB',
    'slateblue':            '#6A5ACD',
    'slategray':            '#708090',
    'snow':                 '#FFFAFA',
    'springgreen':          '#00FF7F',
    'steelblue':            '#4682B4',
    'tan':                  '#D2B48C',
    'teal':                 '#008080',
    'thistle':              '#D8BFD8',
    'tomato':               '#FF6347',
    'turquoise':            '#40E0D0',
    'violet':               '#EE82EE',
    'wheat':                '#F5DEB3',
    'white':                '#FFFFFF',
    'whitesmoke':           '#F5F5F5',
    'yellow':               '#FFFF00',
    'yellowgreen':          '#9ACD32'}


def set_requires_grad(model, requires_grad=True):
    """
    :param model: Instance of Part of Net
    :param requires_grad: Whether Need Gradient
    :return:
    """
    for param in model.parameters():
        param.requires_grad = requires_grad


def fix_randomness(SEED):
    """
    :param SEED:  Random SEED
    :return:
    """
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Evaluator(object):
    def __init__(self, true, pred):
        super(Evaluator, self).__init__()
        self.true = true
        self.pred = pred

    def metric_acc(self):
        return accuracy_score(self.true, self.pred)

    def metric_mf1(self, dataset):
        return f1_score(self.true, self.pred, average="macro")

    def metric_mf1_balance(self):
        return f1_score(self.true, self.pred, average="weighted")

    def metric_confusion_matrix(self):
        return confusion_matrix(self.true, self.pred)

    def classification_report(self):
        return classification_report(self.true, self.pred, target_names=['Sleep stage W',
                                                                         'Sleep stage 1',
                                                                         'Sleep stage 2',
                                                                         'Sleep stage 3/4',
                                                                         'Sleep stage R'])


def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    n_samples = int(source.size()[0])+int(target.size()[0])
    total = torch.cat([source, target], dim=0)

    total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)), int(total.size(2)))

    total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)), int(total.size(2)))

    l2_distance = ((total0-total1)**2).sum(2)

    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(l2_distance.data) / (n_samples**2-n_samples)

    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]

    kernel_val = [torch.exp(-l2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]

    return sum(kernel_val)


def mmd_rbf(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    batch_size = int(source.size()[0])
    kernels = guassian_kernel(source, target, kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    xx = kernels[:batch_size, :batch_size]
    yy = kernels[batch_size:, batch_size:]
    xy = kernels[:batch_size, batch_size:]
    yx = kernels[batch_size:, :batch_size]
    return torch.mean(xx + yy - xy - yx)


class ModelConfig(object):
    def __init__(self, dataset):
        self.dataset = dataset
        self.ConvDrop = 0.1
        self.EncoderParam = EncoderConfig()
        self.SleepMlpParam = SleepMlpParam()
        self.FaceMlpParam = FaceMlpParam()
        self.BCI2000MlpParam = BCI2000MlpParam()
        self.ClassNames = ['W', 'N1', 'N2', 'N3', 'REM']
        self.ClassNamesFace = ['Anger',
                               'Disgust',
                               'Fear',
                               'Sadness',
                               'Neutral',
                               'Amusement',
                               'Inspiration',
                               'Joy',
                               'Tenderness']
        self.ClassNamesBCI2000 = ['Left', 'Right', 'Fist', 'Feet']
        self.ClassNamesTUEV = ["spsw", "gped", "pled", "eyem", "artf", "bckg"]
        self.SeqLength = 20
        self.BatchSize = 32
        self.EpochLength = 3000
        self.EpochLengthFace = 7500
        self.EpochLengthBCI2000 = 640
        channel_num = self.get_channel_info()
        if self.dataset == "ISRUC":
            self.EegNum = channel_num[0]
            self.EogNum = channel_num[1]

        if self.dataset == "FACED":
            self.FaceCn = channel_num[0]

        elif self.dataset == 'BCI2000':
            self.BCICn = channel_num[0]

        if self.dataset == 'ISRUC':
            self.NumClasses = 5
        if self.dataset == 'FACED':
            self.NumClasses = 9
        if self.dataset == 'BCI2000':
            self.NumClasses = 4

    def get_channel_info(self):
        if self.dataset == "ISRUC":
            return [6, 2]
        elif self.dataset == "FACED":
            return [32]
        elif self.dataset == "BCI2000":
            return [64]


class EncoderConfig(object):
    def __init__(self):
        self.n_head = 8
        self.d_model = 512
        self.layer_num = 3
        self.drop = 0.1


class SleepMlpParam(object):
    def __init__(self):
        self.drop = 0.1
        self.first_linear = [512, 256]
        self.second_linear = [256, 128]
        self.out_linear = [128, 5]


class FaceMlpParam(object):
    def __init__(self):
        self.drop = 0.1
        self.first_linear = [512, 256]
        self.second_linear = [256, 128]
        self.out_linear = [128, 9]


class BCI2000MlpParam(object):
    def __init__(self):
        self.drop = 0.1
        self.first_linear = [512, 256]
        self.second_linear = [256, 128]
        self.out_linear = [128, 4]


def compute(li):
    acc_initial = np.mean([i['ACC'][0] for i in (li.values())])

    acc_1 = np.mean([i['ACC'][1] for i in (li.values())])

    acc_after = np.mean([i['ACC'][2] for i in li.values()])

    mf1_initial = np.mean([i['MF1'][0] for i in li.values()])

    mf1_1 = np.mean([i['MF1'][1] for i in li.values()])

    mf1_after = np.mean([i['MF1'][2] for i in li.values()])

    print(f'ACC Before: {acc_initial} '
          f'ACC After{acc_after} '
          f'MF1 Before{mf1_initial} '
          f'MF1 After{mf1_after}')



