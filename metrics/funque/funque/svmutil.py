# TODO: dependency on src/libsvm/svmutil needs to be properly done, this is a temporary workaround wrapper

from __future__ import absolute_import

import sys
from .config import FunqueConfig


libsvm_path = FunqueConfig.root_path('third_party', 'libsvm', 'python')


if libsvm_path not in sys.path:
    # Inject {project}/src/libsvm/python to PYTHONPATH dynamically
    sys.path.append(libsvm_path)


try:
    # This import will work only if above injection was meaningful (ie: user has the files in the right place)
    # from svmutil import *           # noqa
    from libsvm.svmutil import *  # 使用 pip 安装的 libsvm

except ImportError as e:
    print("Can't import svmutil from %s: %s" % (libsvm_path, e))
    sys.exit(1)
