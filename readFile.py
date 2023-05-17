import sys

def readFile(fileName):
    f = open(fileName, "r")
    print(f.read())

readFile(sys.argv[1])