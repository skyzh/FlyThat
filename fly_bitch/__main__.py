from . import data_parser
from . import dataset

if __name__ == '__main__':
    import sys
    if sys.argv[1] == 'data_parser':
        data_parser.main(sys.argv[2:])
    if sys.argv[1] == 'dataset':
        dataset.main(sys.argv[2:])
