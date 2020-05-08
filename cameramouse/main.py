"""
Main file for running the full python computer mouse
"""

import cameramouse
import yaml

with open('config.yaml') as f:
    opts = yaml.load(f, Loader=yaml.FullLoader)

mouse = cameramouse.HandSegmentationMouse(opts)

if __name__ == "__main__":
    mouse.run()
