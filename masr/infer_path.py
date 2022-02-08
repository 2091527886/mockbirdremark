import argparse
from ast import arg
import functools
import time
import wave

from masr.predict import Predictor
from masr.utils.audio_vad import crop_audio_vad
from masr.utils.utils import add_arguments, print_arguments

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg('wav_path',         str,    './dataset/test.wav', "预测音频的路径")
add_arg('is_long_audio',    bool,   False,  "是否为长语音")
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

args = parser.parse_args()
nomoreouputstatus=args.nomoreoutput
if (nomoreouputstatus==0):
    (
    print_arguments(args)
    )



# 获取识别器
predictor = Predictor(model_path=args.model_path, vocab_path=args.vocab_path, use_model=args.use_model,
                      decoder=args.decoder, alpha=args.alpha, beta=args.beta, lang_model_path=args.lang_model_path,
                      beam_size=args.beam_size, cutoff_prob=args.cutoff_prob, cutoff_top_n=args.cutoff_top_n,
                      use_gpu=args.use_gpu, use_pun_model=args.use_pun, pun_model_dir=args.pun_model_dir,
                      feature_method=args.feature_method)


# 长语音识别
def predict_long_audio():
    start = time.time()
    # 分割长音频
    audios_bytes = crop_audio_vad(args.wav_path)
    texts = ''
    scores = []
    # 执行识别
    
    for i, audio_bytes in enumerate(audios_bytes):
        score, text = predictor.predict(audio_bytes=audio_bytes, to_an=args.to_an)
        texts = texts + text if args.use_pun else texts + '，' + text
        scores.append(score)
        if(nomoreouputstatus==0):
            (
            print("第%d个分割音频, 得分: %d, 识别结果: %s" % (i, score, text))
            )
    if(nomoreouputstatus==0):
        (
        print("最终结果，消耗时间：%d, 得分: %d, 识别结果: %s" % (round((time.time() - start) * 1000), sum(scores) / len(scores), texts))
        )
    else:
        (
        print(texts)
        )    
    


# 短语音识别
def predict_audio():
    start = time.time()
    score, text = predictor.predict(audio_path=args.wav_path, to_an=args.to_an)
    if(nomoreouputstatus==0):(
    print("消耗时间：%dms, 识别结果: %s, 得分: %d" % (round((time.time() - start) * 1000), text, score))
    )
    else:(
        print( text)
    )
    


# 实时识别模拟
def real_time_predict_demo():
    state = None
    result = []
    # 识别间隔时间
    interval_time = 1
    CHUNK = 16000 * interval_time
    all_data = []
    # 读取数据
    wf = wave.open(args.wav_path, 'rb')
    data = wf.readframes(CHUNK)
    # 播放
    while data != b'':
        all_data.append(data)
        start = time.time()
        score, text, state = predictor.predict_stream(audio_bytes=data, to_an=args.to_an, init_state_h_box=state)
        result.append(text)
        if(nomoreouputstatus==0):
            (
            print("分段结果：消耗时间：%dms, 识别结果: %s, 得分: %d" % ((time.time() - start) * 1000, ''.join(result), score))
            )
        else:(
            
        )
        
        data = wf.readframes(CHUNK)
    all_data = b''.join(all_data)
    start = time.time()
    score, text, state = predictor.predict_stream(audio_bytes=all_data, to_an=args.to_an, is_end=True)
    if(nomoreouputstatus==0):(
        print("整一句结果：消耗时间：%dms, 识别结果: %s, 得分: %d" % ((time.time() - start) * 1000, text, score))
    )
    else:(
        print(text)
    )
    


if __name__ == "__main__":
    if args.real_time_demo:
        real_time_predict_demo()
    else:
        if args.is_long_audio:
            predict_long_audio()
        else:
            predict_audio()
