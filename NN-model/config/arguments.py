import argparse

def get_arg():
    parser = argparse.ArgumentParser(description='NN-simulator')

    # Add config here
    
    # running config
    parser.add_argument('--use-cuda', action='store_true', default=False,
        help='Use cuda to speed up algorithm.')
    parser.add_argument('--mode', default="normal",
        help='Mode: [normal|calssify] -> normal: output failure impact, classify: output the probability that a failure scenario is a critical failure scenario.')
    parser.add_argument('--log-dir', default="test",
        help='Name of model and log dir.')
    parser.add_argument('--model-load-dir', default=None,
        help='Load trained model from target dir')
    parser.add_argument('--batch-size', type=int, default=25,
        help='Batch size for the model training.') 
    parser.add_argument('--training-epochs', type=int, default=100,
        help='Number of training epochs.')
    parser.add_argument('--train-data-dir', default="./data/train",
        help='Training data directory')
    parser.add_argument('--valid-data-dir', default="./data/valid",
        help='Validation data directory')
    parser.add_argument('--eval-data-dir', default="./data/eval",
        help='Evaluation data directory')
    parser.add_argument('--failure-type', default="multi",
        help='Failure type: [multi|triple]')
    parser.add_argument('--part-failure', action='store_true', default=False,
        help='Select part of failures in training process, only for P2 training of large topologies.')
    parser.add_argument('--detect-normal-model-dir', default=None,
        help='Load trained regression model from target dir for critical failure dection')
    parser.add_argument('--detect-classify-model-dir', default=None,
        help='Load trained classification model from target dir for critical failure dection')
    
    
    # NN model config
    parser.add_argument('--hidden-units', type=int, default=64,
        help='Hidden units of GAT model.')
    parser.add_argument('--global-heads', type=int, default=4, 
        help='Number of global attention heads.')
    parser.add_argument('--local-heads', type=int, default=4, 
        help='Number of local attention heads.')
    parser.add_argument('--node-features-dim', type=int, default=16, 
        help='Input state dimension.')
    
    
    args = parser.parse_args()

    return args