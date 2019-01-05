"""
Python module entry point
"""

import argparse

from . import run

_description = '''Draw Your Own Pinball!

Draw a pinball field on a piece of paper and scan it via webcam.

Rules:

- The ball is lost when it goes off screen.
- You have 999 lives.
- You can place the ball anywhere on the playing field.
- The flippers are controlled by the left and right mouse button.
- Follow the instruction in the console.
- Any bugs are features. 
- Press escape to exit at any points. Do not pass Go. Do not collect $200.
'''

if __name__ == '__main__':
    ap = argparse.ArgumentParser(
        prog='Pinball',
        description=_description,
        epilog='(c) 2018 Dries Kennes',
        formatter_class=argparse.RawTextHelpFormatter,
    )

    ap.add_argument('-f', '--field', required=False, help='Load playing field from image file instead of webcam.')

    args = ap.parse_args()
    run(**vars(args))
