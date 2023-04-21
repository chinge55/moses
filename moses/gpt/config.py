import argparse 

def get_parser(parser = None):
    if parser is None: 
        parser = argparse.ArgumentParser()
    
    #model
    model_arg = parser.add_argument_group('Model')
    model_arg.add_argument("--attn_dropout", type=float, default = 0.1,
                           help = 'Attention Dropout')
    model_arg.add_argument("--embed_dropout", type = float, default = 0.1, 
                           help = "Embedding Dropout")
    model_arg.add_argument("--ff_dropout", type = float, default = 0.1, 
                           help = 'Dropout in the FF layer ')
    
    model_arg.add_argument("--num_heads", type = int, default = 12, 
                           help = "Number of attention Heads")
    model_arg.add_argument("--num_blocks", type = int, default = 12, 
                           help = "Total Decoder blocks used")
    model_arg.add_argument("--embed_dim", type = int, default = 768, 
                           help = "Embedding layer dimensions")
    # Train
    train_arg = parser.add_argument_group('Training')
    train_arg.add_argument('--train_epochs', type=int, default=80,
                           help='Number of epochs for model training')
    train_arg.add_argument('--n_batch', type=int, default=64,
                           help='Size of batch')
    train_arg.add_argument('--lr', type=float, default=1e-3,
                           help='Learning rate')
    train_arg.add_argument('--step_size', type=int, default=10,
                           help='Period of learning rate decay')
    train_arg.add_argument('--gamma', type=float, default=0.5,
                           help='Multiplicative factor of learning rate decay')
    train_arg.add_argument('--n_jobs', type=int, default=1,
                           help='Number of threads')
    train_arg.add_argument('--n_workers', type=int, default=1,
                           help='Number of workers for DataLoaders')

    return parser

def get_config():
    parser = get_parser()
    return parser.parse_known_args()[0]