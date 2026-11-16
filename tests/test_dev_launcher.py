import runpy
import socket
from pathlib import Path
import pytest

require_free_port = runpy.run_path(str(Path(__file__).resolve().parents[1] / 'scripts/dev.py'))['require_free_port']


def test_port_check_rejects_a_real_listener_and_accepts_it_after_close():
    listener = socket.socket()
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(('127.0.0.1', 0))
    port = listener.getsockname()[1]
    listener.listen()
    try:
        with pytest.raises(RuntimeError, match='already in use'):
            require_free_port(port)
    finally:
        listener.close()
    require_free_port(port)
