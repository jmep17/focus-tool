"""Stand-in for the `gh` CLI, put on PATH by test_focus.py.

focus reaches GitHub only by shelling out to gh, so faking the binary is the whole
seam — no network, no auth, and the real argv is exercised. FOCUS_TEST_GH picks the
scenario: "ok" (a PR exists) or "nopr" (the branch has none).
"""
import json
import os
import sys

DIFF = """--- a/payments/client.py
+++ b/payments/client.py
@@ -1,3 +1,9 @@
+def retry(fn):
+    for i in range(3):
+        try:
+            return fn()
+        except TransientError:
+            continue
"""

def main():
    argv = sys.argv[1:]
    if os.environ.get("FOCUS_TEST_GH", "ok") == "nopr":
        sys.stderr.write('no pull requests found for branch "feature"\n')
        return 1
    number = next((int(a) for a in argv[2:] if a.isdigit()), 4521)
    if argv[:2] == ["pr", "view"]:
        print(json.dumps({
            "number": number,
            "title": "Retry payments on transient errors",
            "body": "Implements sc-12345. Retries are capped at three attempts.",
            "url": f"https://github.com/acme/api/pull/{number}",
            "headRefName": "jorden/retry-payments",
            "baseRefName": "main",
            "state": "OPEN",
            "isDraft": False,
        }))
        return 0
    if argv[:2] == ["pr", "diff"]:
        sys.stdout.write(DIFF)
        return 0
    sys.stderr.write("unknown command\n")
    return 1

if __name__ == "__main__":
    sys.exit(main())
