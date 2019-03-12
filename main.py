import argparse
from mlp import MLP
from mlp_aux import MLP_AUX
from parallel import Parallel
from parallel_aux import Parallel_AUX
from dataset import Dataset
from evaluate import evaluate_model

def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='ml-1m',
                        help='Choose a dataset.')
    parser.add_argument('--epochs', type=int, default=8,
                        help='Number of epochs.')
    parser.add_argument('--batch_size', type=int, default=1024,
                        help='Batch size.')
    

    parser.add_argument('--layers',  default='[64,32,16]')

    parser.add_argument('--network_type' , default='mlp', 
                        help='Neural network type. Options are mlp, mlp_aux, parallel, parallel_aux')
    parser.add_argument('--sim_threshold', type=float, default=0.1, 
                        help='The thresold for considering two distinct users similar. Only used in auxiliary learning.')
    parser.add_argument('--negative_sampling_size' , type=int, default = 4,
                        help='Number of zeros sampled for each one.')
    parser.add_argument('--core_number' , type=int, default = 3,
                        help='Number of cores the program will use in its calculations.')
    parser.add_argument('--validation_split' , type=float, default = 0.2,
                        help='Portion of train data considered as validation data')
    parser.add_argument('--user_sampling_size' , type=int, default = 4,
                        help='Number of similar and dissimilar users to sample for each (user,item) tuple. \
                                Only used in auxiliary learning.')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    if args.network_type   == 'mlp':

        dataset = Dataset(dataset_name=args.dataset)
        mlp = MLP(dataset, args.negative_sampling_size, eval(args.layers), 
            args.epochs, args.batch_size, args.validation_split)
        model = mlp.train_model()
        hits, ndcgs = evaluate_model(model, dataset.test_data, dataset.test_negatives, 10, 1)
        print("Hitrate: {}".format(sum(hits) / len(hits)))
        print("NDCG: {}".format(sum(ndcgs) / len(ndcgs)))
    elif args.network_type == 'mlp-aux':

        dataset = Dataset(dataset_name = args.dataset)
        mlp_aux = MLP_AUX(dataset, args.negative_sampling_size, eval(args.layers), 
                    args.epochs, args.batch_size, args.validation_split,
                    args.user_sampling_size, args.core_number, args.sim_threshold)
        model = mlp_aux.train_model()
        hits, ndcgs = evaluate_model(model, dataset.test_data, dataset.test_negatives, 10, 1, True)
        print("Hitrate: {}".format(sum(hits) / len(hits)))
        print("NDCG: {}".format(sum(ndcgs) / len(ndcgs)))
    elif args.network_type == 'parallel':

        dataset = Dataset(dataset_name=args.dataset)
        parallel = Parallel(dataset, args.negative_sampling_size, eval(args.layers), 
            args.epochs, args.batch_size, args.validation_split)
        model = parallel.train_model()
        hits, ndcgs = evaluate_model(model, dataset.test_data, dataset.test_negatives, 10, 1)
        print("Hitrate: {}".format(sum(hits) / len(hits)))
        print("NDCG: {}".format(sum(ndcgs) / len(ndcgs)))
    elif args.network_type == 'parallel-aux':
        dataset = Dataset(dataset_name = args.dataset)
        parallel_aux = Parallel_AUX(dataset, args.negative_sampling_size, eval(args.layers), 
                    args.epochs, args.batch_size, args.validation_split,
                    args.user_sampling_size, args.core_number, args.sim_threshold)
        model = parallel_aux.train_model()
        hits, ndcgs = evaluate_model(model, dataset.test_data, dataset.test_negatives, 10, 1, True)
        print("Hitrate: {}".format(sum(hits) / len(hits)))
        print("NDCG: {}".format(sum(ndcgs) / len(ndcgs)))
    else: 
        print('ERROR: No such network.')