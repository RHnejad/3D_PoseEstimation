import argparse

parser = argparse.ArgumentParser(prog='3DHPE')

parser.add_argument('-s', '--system', choices=["izar","vita17","laptop"])

args = parser.parse_args()
