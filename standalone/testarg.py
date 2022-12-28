import argparse

cand = [0,1,2,3,4,5,6,7]
def print_param(l:list):
    l = [int(i) for i in l]
    print(f'selected: {cand[l]}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--list", '-l', nargs='+')
    args = parser.parse_args()

    print(args.list)
    print(type(args.list))
    print_param(args.list)