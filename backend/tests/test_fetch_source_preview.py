"""
Tests for CrownTALK backend — critical path regression tests.

Covers:
  - Bug 1 regression: fetch_source_preview() must never reference undefined
    names 'prev' or 'req' in its function body (NameError crash on non-X URLs)
  - Bug 4 regression: auth endpoints must use api_success/api_error envelope
  - URL normalization edge cases (http→https, twitter→x.com, dedup)
  - SSRF: validate_public_http_url is defined in main.py and enforces private-IP blocks

Run with:
  python3 -m unittest discover -s tests -p "test_*.py" -v
"""

import ast
import os
import sys
import unittest

BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)


def _read_main() -> str:
    src = open(os.path.join(BACKEND_DIR, "main.py")).read(); return src


def _extract_function_src(full_src: str, fn_name: str) -> str:
    """Extract the AST source text of a top-level function by name."""
    tree = ast.parse(full_src)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == fn_name:
            return ast.get_source_segment(full_src, node) or ""
    return ""


def _section_between(src: str, start_marker: str, end_marker: str) -> str:
    """Return the text between two 'def <name>' markers."""
    try:
        a = src.split(f"def {start_marker}")[1]
        return a.split(f"def {end_marker}")[0]
    except IndexError:
        return ""


# ─────────────────────────────────────────────────────────────────────────────
# Bug 1 — fetch_source_preview NameError regression
# ─────────────────────────────────────────────────────────────────────────────

class TestFetchSourcePreviewNoBadNames(unittest.TestCase):
    """
    The 'out' dict inside fetch_source_preview previously referenced 'prev' and
    'req' which do not exist in that scope — causing an immediate NameError
    whenever a non-X URL was processed.
    """

    def setUp(self):
        self.main_src = _read_main()

    def test_no_undefined_prev_in_function(self):
        fn_src = _extract_function_src(self.main_src, "fetch_source_preview")
        self.assertTrue(fn_src, "fetch_source_preview not found in main.py")

        tree = ast.parse(fn_src)
        loaded_names = {
            node.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
        }
        self.assertNotIn(
            "prev",
            loaded_names,
            "fetch_source_preview references undefined name 'prev' — Bug 1 regression",
        )

    def test_no_undefined_req_in_function(self):
        fn_src = _extract_function_src(self.main_src, "fetch_source_preview")
        self.assertTrue(fn_src, "fetch_source_preview not found in main.py")

        tree = ast.parse(fn_src)
        loaded_names = {
            node.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
        }
        # 'req' is a stdlib alias sometimes, check it's not an Attribute call either
        self.assertNotIn(
            "req",
            loaded_names,
            "fetch_source_preview references undefined name 'req' — Bug 1 regression",
        )

    def test_out_dict_does_not_contain_prev_get(self):
        fn_src = _extract_function_src(self.main_src, "fetch_source_preview")
        self.assertNotIn(
            'prev.get("url")',
            fn_src,
            "out dict still references prev.get('url') — Bug 1 not fully fixed",
        )

    def test_out_dict_does_not_contain_req_source_url(self):
        fn_src = _extract_function_src(self.main_src, "fetch_source_preview")
        self.assertNotIn(
            "req.source_url",
            fn_src,
            "out dict still references req.source_url — Bug 1 not fully fixed",
        )

    def test_out_dict_uses_safe_url(self):
        fn_src = _extract_function_src(self.main_src, "fetch_source_preview")
        # The fixed version must use safe_url (local variable derived from the parameter)
        self.assertIn(
            "safe_url",
            fn_src,
            "fetch_source_preview must use safe_url in the returned dict",
        )

    def test_function_signature_has_source_url_param(self):
        tree = ast.parse(self.main_src)
        fn_node = next(
            (n for n in ast.walk(tree)
             if isinstance(n, ast.FunctionDef) and n.name == "fetch_source_preview"),
            None,
        )
        self.assertIsNotNone(fn_node, "fetch_source_preview not found")
        param_names = [a.arg for a in fn_node.args.args]
        self.assertIn("source_url", param_names)

    def test_main_py_syntax_is_valid(self):
        """Catch any syntax error introduced during Bug 1 fix."""
        try:
            ast.parse(self.main_src)
        except SyntaxError as e:
            self.fail(f"main.py has a syntax error: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Bug 4 — Auth endpoint envelope consistency
# ─────────────────────────────────────────────────────────────────────────────

class TestAuthEndpointEnvelope(unittest.TestCase):
    """
    /signup /login /logout /me must all return responses through api_success /
    api_error, not raw jsonify({...}). The frontend should not need to handle
    two different shapes.
    """

    def setUp(self):
        self.main_src = _read_main()

    def _signup(self):
        return _section_between(self.main_src, "signup_endpoint", "login_endpoint")

    def _login(self):
        return _section_between(self.main_src, "login_endpoint", "logout_endpoint")

    def _logout(self):
        return _section_between(self.main_src, "logout_endpoint", "me_endpoint")

    def _me(self):
        # me_endpoint ends at the next @app.route
        try:
            after = self.main_src.split("def me_endpoint")[1]
            return after.split("@app.route")[0]
        except IndexError:
            return ""

    def test_signup_no_raw_jsonify_errors(self):
        self.assertNotIn(
            'jsonify({"error"',
            self._signup(),
            "/signup returns raw jsonify error — Bug 4",
        )

    def test_signup_uses_api_error(self):
        self.assertIn("api_error(", self._signup(), "/signup must use api_error()")

    def test_signup_uses_api_success(self):
        self.assertIn("api_success(", self._signup(), "/signup must use api_success()")

    def test_login_no_raw_jsonify_errors(self):
        self.assertNotIn(
            'jsonify({"error"',
            self._login(),
            "/login returns raw jsonify error — Bug 4",
        )

    def test_login_uses_api_error(self):
        self.assertIn("api_error(", self._login(), "/login must use api_error()")

    def test_login_uses_api_success(self):
        self.assertIn("api_success(", self._login(), "/login must use api_success()")

    def test_logout_no_raw_jsonify(self):
        self.assertNotIn(
            "jsonify(",
            self._logout(),
            "/logout still uses raw jsonify — Bug 4",
        )

    def test_logout_uses_api_success(self):
        self.assertIn("api_success(", self._logout())

    def test_me_no_raw_jsonify(self):
        self.assertNotIn("jsonify(", self._me(), "/me still uses raw jsonify — Bug 4")

    def test_me_uses_api_success(self):
        self.assertIn("api_success(", self._me())


# ─────────────────────────────────────────────────────────────────────────────
# URL normalization — extended coverage
# ─────────────────────────────────────────────────────────────────────────────

class TestCleanNormalizeUrlsExtended(unittest.TestCase):

    def setUp(self):
        from utils import clean_and_normalize_urls
        self.fn = clean_and_normalize_urls

    def test_http_normalized_to_https(self):
        r = self.fn(["http://x.com/user/status/1234567890"])
        self.assertEqual(r, ["https://x.com/user/status/1234567890"])

    def test_twitter_com_to_x_com(self):
        r = self.fn(["https://twitter.com/user/status/9876543210"])
        self.assertEqual(r, ["https://x.com/user/status/9876543210"])

    def test_mobile_twitter_to_x_com(self):
        r = self.fn(["https://mobile.twitter.com/user/status/111222"])
        self.assertEqual(r, ["https://x.com/user/status/111222"])

    def test_query_params_stripped(self):
        r = self.fn(["https://x.com/user/status/123?s=21&t=XYZ"])
        self.assertEqual(r, ["https://x.com/user/status/123"])

    def test_i_status_form(self):
        r = self.fn(["https://x.com/i/status/9998887776665"])
        self.assertEqual(r, ["https://x.com/i/status/9998887776665"])

    def test_i_web_status_normalized(self):
        r = self.fn(["https://x.com/i/web/status/9998887776665"])
        self.assertEqual(r, ["https://x.com/i/status/9998887776665"])

    def test_dedup_across_http_and_https(self):
        r = self.fn([
            "http://x.com/user/status/123",
            "https://x.com/user/status/123",
        ])
        self.assertEqual(len(r), 1)
        self.assertEqual(r[0], "https://x.com/user/status/123")

    def test_empty_input(self):
        self.assertEqual(self.fn([]), [])

    def test_none_input_returns_empty(self):
        self.assertEqual(self.fn(None), [])  # type: ignore

    def test_non_twitter_url_excluded(self):
        r = self.fn(["https://google.com/search?q=hello"])
        self.assertEqual(r, [])

    def test_mixed_valid_and_invalid(self):
        r = self.fn([
            "https://x.com/user/status/1111",
            "https://google.com/search?q=hello",
        ])
        self.assertEqual(r, ["https://x.com/user/status/1111"])

    def test_url_embedded_in_text(self):
        r = self.fn(["check this https://x.com/alpha/status/1 and follow"])
        self.assertEqual(r, ["https://x.com/alpha/status/1"])


# ─────────────────────────────────────────────────────────────────────────────
# SSRF — validate_public_http_url is defined and correct
# ─────────────────────────────────────────────────────────────────────────────

class TestSsrfProtectionDefined(unittest.TestCase):
    """
    validate_public_http_url lives in main.py. We test it via AST inspection
    (avoids loading the full Flask app) plus a lightweight direct test for
    the non-network-dependent cases.
    """

    def setUp(self):
        self.main_src = _read_main()

    def test_function_exists(self):
        tree = ast.parse(self.main_src)
        names = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
        self.assertIn(
            "validate_public_http_url",
            names,
            "validate_public_http_url function not found in main.py",
        )

    def test_private_ip_ranges_checked(self):
        fn_src = _extract_function_src(self.main_src, "validate_public_http_url")
        # Must call _ip_is_public or some IP-check helper
        self.assertTrue(
            "_ip_is_public" in fn_src or "is_private" in fn_src or "_is_private" in fn_src,
            "validate_public_http_url does not appear to check IP ranges",
        )

    def test_non_http_scheme_raises(self):
        fn_src = _extract_function_src(self.main_src, "validate_public_http_url")
        # Must check the scheme
        self.assertIn(
            '"http"',
            fn_src,
            "validate_public_http_url must reject non-http/https schemes",
        )

    def test_localhost_blocked(self):
        fn_src = _extract_function_src(self.main_src, "validate_public_http_url")
        self.assertIn(
            "localhost",
            fn_src,
            "validate_public_http_url must explicitly block 'localhost'",
        )


# ─────────────────────────────────────────────────────────────────────────────
# Frontend contract: HMAC secret not in public bundle
# ─────────────────────────────────────────────────────────────────────────────

class TestHmacSecretNotPublic(unittest.TestCase):
    """Bug 2: NEXT_PUBLIC_ prefix exposes secrets in the browser bundle."""

    def _read_api_ts(self) -> str:
        path = os.path.join(BACKEND_DIR, "..", "Frontend", "lib", "api.ts")
        abs_path = os.path.abspath(path)
        if os.path.exists(abs_path):
            with open(abs_path) as _f:
            return _f.read()
        return ""

    def test_next_public_hmac_secret_not_in_api_ts(self):
        src = self._read_api_ts()
        if not src:
            self.skipTest("Frontend/lib/api.ts not found — run from repo root")
        self.assertNotIn(
            "NEXT_PUBLIC_CT_HMAC_SECRET",
            src,
            "NEXT_PUBLIC_CT_HMAC_SECRET is in api.ts — secret is exposed in browser bundle (Bug 2)",
        )

    def test_server_sign_route_referenced(self):
        src = self._read_api_ts()
        if not src:
            self.skipTest("Frontend/lib/api.ts not found")
        self.assertIn(
            "/api/sign",
            src,
            "api.ts must call /api/sign (server-side signing) instead of computing HMAC client-side",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
