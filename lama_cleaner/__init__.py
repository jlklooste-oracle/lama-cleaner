from parse_args import parse_args
from server import main


def entry_point():
    print('__init__.py')
    args = parse_args()
    main(args)

args = parse_args()
main(args)
