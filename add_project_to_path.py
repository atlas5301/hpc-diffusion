# project/add_project_to_path.py

import os
import sys

# Find the absolute path of the 'project' directory
project_dir = os.path.dirname(os.path.abspath(__file__))

# Add the 'project' directory to sys.path
if project_dir not in sys.path:
    sys.path.append(project_dir)
