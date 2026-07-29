from scripts import *
from scripts.recolectar_resenas import consolidar_resenas


def main():
     cons = consolidar_resenas('data/raw')
     cons.consolidar()
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
