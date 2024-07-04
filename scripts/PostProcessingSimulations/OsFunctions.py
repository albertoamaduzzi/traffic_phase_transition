import os

def JoinDir(BaseDir,TreeDir):
    return os.path.join(BaseDir,TreeDir)
def IsFile(File):
    return os.path.isfile(File)

def MakeDirs(Dir):
    os.makedirs(Dir,exist_ok=True)
    return Dir