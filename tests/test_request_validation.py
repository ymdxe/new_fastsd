import unittest

from src.request_validation import canonicalize_request


class RequestValidationTests(unittest.TestCase):
    def test_prefill_canonicalizes_gamma_to_zero(self):
        request = canonicalize_request(
            {
                "task_type": "prefill",
                "prefix_len": 3,
                "draft_output": [11, 12, 13],
            }
        )
        self.assertEqual(request["prefix_len"], 3)
        self.assertEqual(request["gamma"], 0)

    def test_verify_requires_positive_gamma_and_payload(self):
        with self.assertRaises(ValueError):
            canonicalize_request(
                {
                    "task_type": "verify",
                    "prefix_len": 3,
                    "draft_output": [11, 12, 13],
                    "gamma": 0,
                }
            )
        with self.assertRaises(ValueError):
            canonicalize_request(
                {
                    "task_type": "verify",
                    "prefix_len": 3,
                    "draft_output": [11, 12, 13],
                    "gamma": 2,
                }
            )

    def test_tail_only_accounts_for_bridge_token(self):
        request = canonicalize_request(
            {
                "task_type": "verify",
                "prefix_len": 8,
                "draft_output": [99, 101, 102],
                "gamma": 2,
                "tail_only": True,
                "has_bridge_token": True,
            }
        )
        self.assertEqual(request["gamma"], 2)
        with self.assertRaises(ValueError):
            canonicalize_request(
                {
                    "task_type": "verify",
                    "prefix_len": 8,
                    "draft_output": [101, 102],
                    "gamma": 2,
                    "tail_only": True,
                    "has_bridge_token": True,
                }
            )

    def test_prefix_bounds_are_rejected(self):
        for prefix_len in (0, -1, 10):
            with self.subTest(prefix_len=prefix_len):
                with self.assertRaises(ValueError):
                    canonicalize_request(
                        {
                            "task_type": "prefill",
                            "prefix_len": prefix_len,
                            "draft_output": [1, 2, 3],
                        }
                    )


if __name__ == "__main__":
    unittest.main()
