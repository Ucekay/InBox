import argparse

def parse_args(args=None):
    parser = argparse.ArgumentParser(description="I2B")

    # ===== environment ===== #
    parser.add_argument("--cuda", action='store_true', help="use gpu or not")
    parser.add_argument("--gpu_id", type=int, default=0, help="gpu id")
    parser.add_argument("--cpu_num", type=int, default=0, help="cpu num for DataLoader workers (0=main process only, recommended for WSL)")

    # ===== dataset ===== #
    parser.add_argument("--dataset", nargs="?", default="last-fm", help="Choose a dataset:[last-fm,amazon-book,alibaba]")
    parser.add_argument("--data_path", nargs="?", default="data/", help="Input data path.")

    # ===== train setting ===== #
    parser.add_argument('-pre', '--do_pretrain', default=False, action='store_true', help="do pretrain")
    parser.add_argument('-pre_i', '--do_pretrain_inter', default=False, action='store_true', help="do pretrain of intersection")
    parser.add_argument('-train', '--do_train', default=False, action='store_true', help="do train")

    parser.add_argument('-pre_test', '--pretrain_do_test', default=False, action='store_true', help="do pretrain test")
    parser.add_argument('-pre_i_test', '--pretrain_inter_do_test', default=False, action='store_true', help="do pretrain intersection test")
    parser.add_argument('-test', '--train_do_test', default=False, action='store_true', help="do test")
    
    parser.add_argument('-valid', '--do_valid', default=False, action='store_true', help="do valid")

    parser.add_argument('-pre_bs', '--pretrain_batch_size', type=int, default=256, help='pretrain batch size')
    parser.add_argument('-pre_i_bs', '--pretrain_inter_batch_size', type=int, default=32, help='pretrain intersection batch size')
    parser.add_argument('-bs', '--train_batch_size', type=int, default=128, help='recommender train batch size')
    
    parser.add_argument('-pre_n', '--pretrain_negative_sample_size', type=int, default=256, help='pretrain negative sample batch size')
    parser.add_argument('-pre_i_n', '--pretrain_inter_negative_sample_size', type=int, default=256, help='pretrain intersection negative sample batch size')
    parser.add_argument('-n', '--train_negative_sample_size', type=int, default=256, help='negative sample batch size')

    parser.add_argument('-pre_test_bs', '--pretrain_test_batch_size', type=int, default=10, help='pretrain batch size')
    parser.add_argument('-pre_i_test_bs', '--pretrain_inter_test_batch_size', type=int, default=10, help='pretrain inter batch size')
    parser.add_argument('-test_bs', '--train_test_batch_size', type=int, default=10, help='batch size')

    parser.add_argument('-pre_lr', '--pretrain_learning_rate', type=float, default=1e-4, help='pretrain learning rate')
    parser.add_argument('-pre_i_lr', '--pretrain_inter_learning_rate', type=float, default=1e-4, help='pretrain intersection learning rate')
    parser.add_argument('-lr', '--train_learning_rate', type=float, default=1e-4, help='learning rate')
    
    parser.add_argument('-pre_epoch', '--pretrain_epoch', type=int, default=80, help='pretrain epoch')
    parser.add_argument('-pre_i_epoch', '--pretrain_inter_epoch', type=int, default=100, help='pretrain intersection epoch')
    parser.add_argument('-epoch', '--train_epoch', type=int, default=30, help='train epoch')

    parser.add_argument('-pre_test_epoch', '--pretrain_test_epoch', type=int, default=40, help='pretrain test pre epoch')
    parser.add_argument('-pre_i_test_epoch', '--pretrain_inter_test_epoch', type=int, default=50, help='pretrain intersection test pre epoch')
    parser.add_argument('-test_epoch', '--test_epoch', type=int, default=3, help='train test pre epoch')

    
    # ===== model ===== #
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('-d', '--dim', type=int, default=512, help='embedding size')
    parser.add_argument("--interest_num", type=int, default=1, help="number of latent factor for user favour")
    parser.add_argument('--l2', type=float, default=1e-5, help='l2 regularization weight')
    parser.add_argument('-g', '--gamma', default=12.0, type=float)
    parser.add_argument('-boxm', '--box_mode', default="(relu, 1.00)", type=str, help='(offset activation,center_reg), center_reg balances the in_box dist and out_box dist')

    # ===== save model ===== #
    parser.add_argument("--save", type=bool, default=True, help="save model or not")
    parser.add_argument("--save_step", type=int, default=10000, help="save model frequence")
    parser.add_argument("--log_step", type=int, default=1000, help="log the metrics frequence")
    # ===== load model ===== #
    parser.add_argument('--checkpoint_path', default=None, type=str)
    parser.add_argument('--checkpoint', default=None, type=str)
    parser.add_argument('--print_on_screen', action='store_true')

    
    return parser.parse_args(args)