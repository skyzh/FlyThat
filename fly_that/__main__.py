from . import data_parser
from . import dataset
from . import model
from . import train
from . import evaluate
from . import focal_loss

if __name__ == '__main__':
    import sys
    if sys.argv[1] == 'data_parser':
        data_parser.main(sys.argv[2:])
    if sys.argv[1] == 'dataset':
        dataset.main(sys.argv[2:])
    if sys.argv[1] == 'model':
        model.main(sys.argv[2:])
    if sys.argv[1] == 'train':
        train.main(sys.argv[2:])
    if sys.argv[1] == 'evaluate':
        evaluate.main(sys.argv[2:])
    if sys.argv[1] == 'loss':
        focal_loss.main(sys.argv[2:])
