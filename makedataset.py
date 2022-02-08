import argparse
from ast import arg
import functools

from matplotlib.pyplot import get
from masr.masr import infer_path
from masr.masr.predict import Predictor
from masr.masr.utils.utils import add_arguments, print_arguments
parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg('wav_path',         str,    './dataset/test.wav', "预测音频的路径")
add_arg('real_time_demo',   bool,   False,  "是否使用实时语音识别演示")
add_arg('use_gpu',          bool,   True,   "是否使用GPU预测")
add_arg('to_an',            bool,   False,  "是否转为阿拉伯数字")
add_arg('use_pun',          bool,   False,  "是否给识别结果加标点符号")
add_arg('nomoreoutput',     bool,   False,  "是否太多输出")
add_arg('beam_size',        int,    300,    "集束搜索解码相关参数，搜索的大小，范围建议:[5, 500]")
add_arg('alpha',            float,  2.2,    "集束搜索解码相关参数，LM系数")
add_arg('beta',             float,  4.3,    "集束搜索解码相关参数，WC系数")
add_arg('cutoff_prob',      float,  0.99,   "集束搜索解码相关参数，剪枝的概率")
add_arg('cutoff_top_n',     int,    40,     "集束搜索解码相关参数，剪枝的最大值")
add_arg('use_model',        str,    'deepspeech2',               "所使用的模型")
add_arg('vocab_path',       str,    'dataset/vocabulary.txt',    "数据集的词汇表文件路径")
add_arg('pun_model_dir',    str,    'models/pun_models/',        "加标点符号的模型文件夹路径")
add_arg('model_path',       str,    'models/deepspeech2/inference.pt', "导出的预测模型文件路径")
add_arg('lang_model_path',  str,    'lm/zh_giga.no_cna_cmn.prune01244.klm',   "集束搜索解码相关参数，语言模型文件路径")
add_arg('feature_method',   str,    'linear',             "音频预处理方法", choices=['linear', 'mfcc', 'fbank'])
add_arg('decoder',          str,    'ctc_beam_search',    "结果解码方法",   choices=['ctc_beam_search', 'ctc_greedy'])
add_arg('output',           str,    '233333333333333',    "结果解码方法",)
print ("-------------------------------------")
args = parser.parse_args()

print_arguments(args)

print ("-------------------------------------")
predictor = Predictor(model_path=args.model_path, vocab_path=args.vocab_path, use_model=args.use_model,
                      decoder=args.decoder, alpha=args.alpha, beta=args.beta, lang_model_path=args.lang_model_path,
                      beam_size=args.beam_size, cutoff_prob=args.cutoff_prob, cutoff_top_n=args.cutoff_top_n,
                      use_gpu=args.use_gpu, use_pun_model=args.use_pun, pun_model_dir=args.pun_model_dir,
                      feature_method=args.feature_method)

def get1(x):
    score, text = predictor.predict(audio_path=x, to_an=args.to_an)
    return(text)
wavpath=args.wav_path
pathtemp=""
#{
# 
# 
# 
# 
# 
# 
# 
# 
# open ....
# 
# from wav path
# 
# 
# 
# 
# 
# 
# 
# 
# 
# }
returnback=""

for x in wavpath:
    pathtemp=x
    returnback = get1(pathtemp)
    #/n






