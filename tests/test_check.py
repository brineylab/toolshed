import unittest, tempfile, sys, io, json, urllib.error
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import check  # noqa: E402

SKILL = "---\nname: {name}\ndescription: A tool.\nmetadata:\n  version: {version}\n---\n# {name}\n"

def tool(root: Path, name="esm2-650m", version="1.0.0", skill=True, containerfile="FROM x\nCOPY test /toolshed/test\n", run=True):
    d = root / "tools" / name
    (d / "test").mkdir(parents=True)
    if skill: (d / "SKILL.md").write_text(SKILL.format(name=name, version=version))
    if containerfile is not None: (d / "Containerfile").write_text(containerfile)
    if run: (d / "test" / "run.sh").write_text("set -euo pipefail\necho ok\n")
    return d

class CheckTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
    def test_frontmatter_parses_name_description_and_nested_version(self):
        d = tool(self.root)
        self.assertEqual(check.frontmatter(d / "SKILL.md"), {"name": "esm2-650m", "description": "A tool.", "metadata": {"version": "1.0.0"}})
    def test_a_good_tool_passes(self):
        self.assertEqual(check.check_tool(tool(self.root), published=set()), [])
    def test_every_rule_names_its_reason(self):
        cases = [
            (dict(name="Esm2"), "name"),                       # directory name not lowercase-hyphen
            (dict(version="1.0"), "metadata.version"),          # not semver
            (dict(skill=False), "SKILL.md"),
            (dict(containerfile=None), "Containerfile"),
            (dict(containerfile="FROM x\n"), "COPY test /toolshed/test"),
            (dict(run=False), "test/run.sh"),
        ]
        for kw, word in cases:
            root = Path(tempfile.mkdtemp())
            reasons = check.check_tool(tool(root, **kw), published=set())
            self.assertTrue(any(word in r for r in reasons), (kw, reasons))
    def test_a_skill_whose_name_is_not_the_directorys_fails(self):
        d = tool(self.root)
        (d / "SKILL.md").write_text(SKILL.format(name="other", version="1.0.0"))
        self.assertTrue(any("name" in r for r in check.check_tool(d, published=set())))
    def test_a_published_version_is_refused(self):
        reasons = check.check_tool(tool(self.root), published={"1.0.0"})
        self.assertEqual(reasons, ["version 1.0.0 is already published; bump metadata.version"])

class Response(io.BytesIO):
    def __init__(self, body, link=None):
        super().__init__(json.dumps(body).encode())
        self.headers = {"Link": link} if link else {}

class PublishedTagsTests(unittest.TestCase):
    def stub(self, fake):
        real = check.urllib.request.urlopen
        check.urllib.request.urlopen = fake
        self.addCleanup(setattr, check.urllib.request, "urlopen", real)
    def test_every_page_of_tags_is_read(self):
        pages = {"/v2/o/r/t/tags/list?n=1000": Response({"tags": ["1.0.0"]}, '</v2/o/r/t/tags/list?last=1.0.0&n=1000>; rel="next"'),
                 "/v2/o/r/t/tags/list?last=1.0.0&n=1000": Response({"tags": ["2.0.0"]})}
        def fake(req):
            url = req if isinstance(req, str) else req.full_url
            return Response({"token": "x"}) if "/token?" in url else pages[url.removeprefix("https://ghcr.io")]
        self.stub(fake)
        self.assertEqual(check.published_tags("ghcr.io/o/r", "t"), {"1.0.0", "2.0.0"})
    def test_a_package_that_does_not_exist_has_no_tags(self):
        def fake(req):  # GHCR refuses an anonymous token for a package that does not exist
            raise urllib.error.HTTPError(str(req), 403, "Forbidden", {}, None)
        self.stub(fake)
        self.assertEqual(check.published_tags("ghcr.io/o/r", "t"), set())
    def test_a_bad_name_is_reported_without_asking_the_registry(self):
        def fake(req):
            raise AssertionError("registry asked about an invalid name")
        self.stub(fake)
        d = tool(Path(tempfile.mkdtemp()), name="Esm2")
        out = io.StringIO()
        argv, stdout = sys.argv, sys.stdout
        sys.argv, sys.stdout = ["check.py", str(d), "--registry", "ghcr.io/o/r"], out
        try:
            code = check.main()
        finally:
            sys.argv, sys.stdout = argv, stdout
        self.assertEqual(code, 1)
        self.assertIn("tools/Esm2: name", out.getvalue())

if __name__ == "__main__":
    unittest.main()
