import sys
if sys.prefix == '/home/nano/.venvs/snail-bottle':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/nano/r20_ws/install/snail_bottle_approach'
