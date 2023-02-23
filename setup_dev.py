from sys import argv
from os import symlink, unlink
from os.path import join, islink
from pathlib import Path
from site import USER_SITE


# Package information
PROJECT = 'vedoTk'
root = join(Path(__file__).parent.absolute(), 'src')

# Check user entry
if len(argv) == 2 and argv[1] not in ['set', 'del']:
    raise ValueError(f"\nInvalid script option."
                     f"\nRun 'python3 dev.py set' to link {PROJECT} to your site package folder."
                     f"\nRun 'python3 dev.py del' to remove {PROJECT} link from your site package folder.")

# Option 1: create the symbolic links
if len(argv) == 1 or argv[1] == 'set':
    if not islink(join(USER_SITE, PROJECT)):
        symlink(src=join(root),
                dst=join(USER_SITE, PROJECT))
        print(f"Linked {join(USER_SITE, PROJECT)} -> {join(root)}")

# Option 2: remove the symbolic links
else:
    if islink(join(USER_SITE, PROJECT)):
        unlink(join(USER_SITE, PROJECT))
        print(f"Unlinked {join(USER_SITE, PROJECT)} -> {join(root)}")
