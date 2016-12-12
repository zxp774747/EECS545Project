import tensorflow as tf
import argparse
import sys
from datetime import datetime
import os.path

def parseArgs():
 
  parser = argparse.ArgumentParser()
  parser.add_argument('--mode', choices=['train', 'sample'], type=str, default='train', help='training mode or sampling mode')
  parser.add_argument('--log_dir', type=str, default='./logs')
  parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints')
  parser.add_argument('--expID', type=str, default='experiment-%s' % str(datetime.now()))
  parser.add_argument('--dataset', choices=['mnist', 'stl10'], default='mnist', type=str, help='the dataset to use')
  parser.add_argument('--checkpoint', type=str, help='model checkpoint path')
  parser.add_argument('--output', type=str, default='results.png', help='visualization output')
  parser.add_argument('--img_height', type=int)
  parser.add_argument('--img_width', type=int)
  parser.add_argument('--n_epochs', type=int, default=1, help='the number of training epochs')
  parser.add_argument('--batch_size', type=int, default=64)
  parser.add_argument('--z_dim', type=int, default=100, help='the dimension of z')
  parser.add_argument('--g_feature_dim', type=int, default=64, help='the input feature dimension of the last conv layer in G')
  parser.add_argument('--d_feature_dim', type=int, default=64, help='the output feature dimension of the first conv layer in G')
  parser.add_argument('--n_threads', type=int, default=4, help='the number of data-loading threads')
  parser.add_argument('--learning_rate', type=float, default=2e-4)
  parser.add_argument('--beta1', type=float, default=0.5)

  args = parser.parse_args()

  if args.mode == 'sample' and (not args.checkpoint or not args.output):
    print '--checkpoint and --output is compulsory when mdoe == sample'
    sys.exit(1)

  
  args.N = 100000 if args.dataset == 'stl10' else 60000  

  if not args.img_height:
    args.img_height = 80 if args.dataset == 'stl10' else 32
  if not args.img_width:
    args.img_width = 80 if args.dataset == 'stl10' else 32
  args.img_depth = 3 if args.dataset == 'stl10' else 1
  assert args.img_height % 16 == 0 and args.img_width % 16 == 0

  args.log_dir = os.path.join(args.log_dir, args.expID)
  args.checkpoint_dir = os.path.join(args.checkpoint_dir, args.expID)
  tf.logging.set_verbosity(tf.logging.INFO)
  if mode == 'train' and not tf.gfile.Exists(args.log_dir):
    tf.gfile.MakeDirs(args.log_dir)
  if mode == 'train' and not tf.gfile.Exists(args.checkpoint_dir):
    tf.gfile.MakeDirs(args.checkpoint_dir)

  return args
