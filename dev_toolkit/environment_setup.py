"""
    The file is using Django style idea of maintaining the root project path.
    These two variables can be used in the files in order to access other files using relative paths
"""
import os

PROJECT_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
LOGDIR_PREFIX = 'execution_results'
