import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="NSAP testing for the recommendation dataset")

    #============train=============#
    parser.add_argument("--gpu-id", type=int, default=0, help="gpu id")
    parser.add_argument("--cuda", type=bool, default=True, help="use gpu or not")
    parser.add_argument('--num-ntype', type=int, default=4, help='the number of types of nodes')

    parser.add_argument('--repeat', type=int, default=3,
                        help='Repeat the training and testing for N times. Default is 1.')
    parser.add_argument('--lr', type=float, default=0.005, help='learning rate.Default is 1e-2.')
    parser.add_argument('--weight-decay',type=float,default=1e-3, help="weight decay.Default is 1e-3.")
    parser.add_argument('--patience', type=int, default=5, help='Patience. Default is 4.')
    parser.add_argument('--save-postfix', default='LastNSAP', help='Postfix for the saved model and result. Default is LastNSAP.')
    parser.add_argument('--epoch', type=int, default=50, help='Number of epochs. Default is 50.')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size. Default is 8.')
    parser.add_argument('--samples', type=int, default=150, help='Number of neighbors sampled. Default is 100.')
    parser.add_argument('--num-heads', type=int, default=4, help='Number of the attention heads. Default is 1.')
    parser.add_argument('--in-size', type=int, default=1024, help='Dimension of the node in state. Default is 1024.')
    parser.add_argument('--hidden-size', type=int, default=128, help='Dimension of the node hidden state. Default is 64.')
    parser.add_argument('--out-size', type=int, default=64, help='Dimension of the node out state. Default is 64.')
    parser.add_argument('--dropout-rate', type=float, default=0.5, help='drop out rate. Default is 0.5.')
    parser.add_argument('-ld', '--log-dir', type=str, default='results',help='Dir for saving training results')

    return parser.parse_args()

