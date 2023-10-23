import os
import sys
import json
def ifnotexistsmkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)
    return path
class configuration:
    def __init__(self):
        self.working_dir = sys.path[0] 



    def dump_configuration(self):
        dumping_dir = ifnotexistsmkdir(os.path.join(self.working_dir,'configuration'))
        with open(dumping_dir,'w') as f:
            json.dumps(f,indent = 4)