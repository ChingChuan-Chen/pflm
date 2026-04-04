#!/usr/bin/env python3


# Authors: Ching-Chuan Chen
# SPDX-License-Identifier: MIT

import os

pflm_init = os.path.join(os.path.dirname(__file__), "../__init__.py")

data = open(pflm_init).readlines()
version_line = next(line for line in data if line.startswith("__version__"))

version = version_line.strip().split(" = ")[1].replace('"', "").replace("'", "")

print(version)
