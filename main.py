from SourceFiles.beamreconstruction import *
import sys

def main():
    if (len(sys.argv) != 3):
        print("usage: python3 main.py <mode> <root file name | all>")
        sys.exit()

    root_file_name = "RootFiles/" + sys.argv[1]

    beam_reconstruction(root_file_name)

if __name__ == "__main__":
    main()
