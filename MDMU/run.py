import argparse
from utils.en_train import EnConfig, EnRun
from utils.ch_train import ChConfig, ChRun
from distutils.util import strtobool

def main(args):
    if args.dataset != 'SIMS':
        EnRun(EnConfig(batch_size=args.batch_size,learning_rate=args.lr,seed=args.seed, model=args.model, tasks = args.tasks
                       , dataset_name=args.dataset,context=args.context, text_context_len=args.text_context_len, audio_context_len=args.audio_context_len,
                       video_context_len=args.video_context_len))
    else:
        ChRun(ChConfig(batch_size=args.batch_size,learning_rate=args.lr,seed=args.seed, model=args.model, tasks = args.tasks
                       , dataset_name=args.dataset,context=args.context, text_context_len=args.text_context_len, audio_context_len=args.audio_context_len,
                       video_context_len=args.video_context_len))
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size')
    parser.add_argument('--lr', type=float, default=5e-6, help='learning rate, recommended: 5e-6 for MOSI, mosei, 1e-5 for sims')
    parser.add_argument('--model', type=str, default='cc', help='concatenate(cc) or cross-modality encoder(cme)')
    parser.add_argument('--dataset', type=str, default='MOSI', help='dataset name: MOSI, MOSEI, SIMS')
    parser.add_argument('--tasks', type=str, default='M', help='losses to train: M: multi-modal, T: text, A: audio (defalut: MTA))')
    parser.add_argument('--context', default=True, help='incorporate context or not', dest='context', type=lambda x: bool(strtobool(x)))
    parser.add_argument('--text_context_len', type=int, default=1)
    parser.add_argument('--audio_context_len', type=int, default=1)
    parser.add_argument('--video_context_len', type=int, default=1)
    args = parser.parse_args()
    main(args)





