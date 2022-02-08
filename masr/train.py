import argparse
import functools

from masr.trainer import MASRTrainer
from masr.utils.utils import add_arguments, print_arguments

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg('batch_size',       int,    16,                       '训练的批量大小')
add_arg('local_rank',       int,    -1,                       '多卡训练的节点')
add_arg('num_workers',      int,    8,                        '读取数据的线程数量')
add_arg('num_epoch',        int,    65,                       '训练的轮数')
add_arg('learning_rate',    int,    5e-5,                     '初始学习率的大小')
add_arg('min_duration',     int,    0.5,                      '过滤最短的音频长度')
add_arg('max_duration',     int,    20,                       '过滤最长的音频长度，当为-1的时候不限制长度')
add_arg('alpha',            float,  2.2,                      '集束搜索的LM系数')
add_arg('beta',             float,  4.3,                      '集束搜索的WC系数')
add_arg('beam_size',        int,    300,                      '集束搜索的大小，范围:[5, 500]')
add_arg('num_proc_bsearch', int,    10,                       '集束搜索方法使用CPU数量')
add_arg('cutoff_prob',      float,  0.99,                     '剪枝的概率')
add_arg('cutoff_top_n',     int,    40,                       '剪枝的最大值')
add_arg('use_model',        str,    'deepspeech2',              '所使用的模型')
add_arg('train_manifest',   str,    'dataset/manifest.train',   '训练数据的数据列表路径')
add_arg('test_manifest',    str,    'dataset/manifest.test',    '测试数据的数据列表路径')
add_arg('dataset_vocab',    str,    'dataset/vocabulary.txt',   '数据字典的路径')
add_arg('mean_std_path',    str,    'dataset/mean_std.npz',     '数据集的均值和标准值的npy文件路径')
add_arg('augment_conf_path',str,    'conf/augmentation.json',   '数据增强的配置文件，为json格式')
add_arg('save_model_path',  str,    'models/',                  '模型保存的路径')
add_arg('metrics_type',     str,    'cer',                      '计算错误率方法', choices=['cer', 'wer'])
add_arg('feature_method',   str,    'linear',                   '音频预处理方法', choices=['linear', 'mfcc', 'fbank'])
add_arg('decoder',          str,    'ctc_greedy',               '结果解码方法', choices=['ctc_beam_search', 'ctc_greedy'])
add_arg('lang_model_path',  str,    'lm/zh_giga.no_cna_cmn.prune01244.klm',        "语言模型文件路径")
add_arg('resume_model',     str,    None,                       '恢复训练，当为None则不使用预训练模型')
add_arg('pretrained_model', str,    None,                       '预训练模型的路径，当为None则不使用预训练模型')
args = parser.parse_args()
print_arguments(args)

trainer = MASRTrainer(use_model=args.use_model,
                      feature_method=args.feature_method,
                      mean_std_path=args.mean_std_path,
                      train_manifest=args.train_manifest,
                      test_manifest=args.test_manifest,
                      dataset_vocab=args.dataset_vocab,
                      num_workers=args.num_workers,
                      alpha=args.alpha,
                      beta=args.beta,
                      beam_size=args.beam_size,
                      num_proc_bsearch=args.num_proc_bsearch,
                      cutoff_prob=args.cutoff_prob,
                      cutoff_top_n=args.cutoff_top_n,
                      decoder=args.decoder,
                      metrics_type=args.metrics_type,
                      lang_model_path=args.lang_model_path)

trainer.train(batch_size=args.batch_size,
              min_duration=args.min_duration,
              max_duration=args.max_duration,
              num_epoch=args.num_epoch,
              learning_rate=args.learning_rate,
              save_model_path=args.save_model_path,
              resume_model=args.resume_model,
              pretrained_model=args.pretrained_model,
              augment_conf_path=args.augment_conf_path)
