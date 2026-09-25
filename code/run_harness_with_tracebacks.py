"""Run the unchanged harness CLI with periodic stack dumps for stalled workers."""
import faulthandler
import os
import runpy

faulthandler.enable()
faulthandler.dump_traceback_later(int(os.environ.get('VLA_TRACEBACK_TIMEOUT', '120')), repeat=True)
try:
    if os.environ.get('VLA_FAST_GLASS_BLUR') == '1':
        from fast_glass_blur import install
        install()
    runpy.run_module('vla_eval.cli.main', run_name='__main__', alter_sys=True)
finally:
    faulthandler.cancel_dump_traceback_later()
