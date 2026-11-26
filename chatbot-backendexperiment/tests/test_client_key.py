"""Rate-limit client key: proxy headers are honoured only when trusted."""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

# Ensure backend root is on sys.path
BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from starlette.requests import Request

from app import config, main


def _request(headers=None, client=("10.0.0.1", 1234)):
    raw = [(k.lower().encode(), v.encode()) for k, v in (headers or {}).items()]
    return Request({"type": "http", "headers": raw, "client": client, "query_string": b""})


class ClientKeyTests(unittest.TestCase):
    def test_ignores_proxy_headers_by_default(self):
        with patch.object(config, "TRUST_PROXY_HEADERS", False):
            req = _request({"x-forwarded-for": "1.2.3.4", "x-real-ip": "5.6.7.8"})
            self.assertEqual(main._client_key(req), "10.0.0.1")

    def test_uses_real_ip_when_trusted(self):
        with patch.object(config, "TRUST_PROXY_HEADERS", True):
            req = _request({"x-forwarded-for": "1.2.3.4", "x-real-ip": "5.6.7.8"})
            self.assertEqual(main._client_key(req), "5.6.7.8")

    def test_uses_first_forwarded_hop_when_trusted(self):
        with patch.object(config, "TRUST_PROXY_HEADERS", True):
            req = _request({"x-forwarded-for": "1.2.3.4, 9.9.9.9"})
            self.assertEqual(main._client_key(req), "1.2.3.4")

    def test_falls_back_to_socket_peer_when_no_headers(self):
        with patch.object(config, "TRUST_PROXY_HEADERS", True):
            self.assertEqual(main._client_key(_request()), "10.0.0.1")

    def test_unknown_without_any_client(self):
        with patch.object(config, "TRUST_PROXY_HEADERS", False):
            self.assertEqual(main._client_key(_request(client=None)), "unknown")


if __name__ == "__main__":
    unittest.main()
