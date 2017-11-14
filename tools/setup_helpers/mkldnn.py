import platform

from .env import check_env_flag

WITH_MKLDNN = not check_env_flag('NO_MKLDNN') and platform.system() == 'Linux'
WITH_AVX512 = not check_env_flag('NO_AVX512')
