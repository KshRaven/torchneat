"""Project-specific settings for pypimgr in this repository."""

PROJECT_NAME = "torchneat"
PYPI_PACKAGE_NAME = "torchneat"
GIT_REMOTE = "origin"
TAG_PREFIX = "v"
RELEASE_COMMIT_TEMPLATE = "Release v{version}"

# Project-relative paths.
PROJECT_FILE = "pyproject.toml"
CHANGELOG_FILE = "CHANGELOG.md"
ENV_FILE = ".env.release"
COMMIT_MESSAGE_FILE = "commit.txt"
REQUIRED_FILES = (
    "README.md",
    "LICENSE",
    PROJECT_FILE,
    CHANGELOG_FILE,
    "MANIFEST.in",
    ".gitignore",
)
# Directory names omitted from the source staging copy and regexes checked against archive paths.
TOP_LEVEL_EXCLUDES = ("data",)
EXTRA_FORBIDDEN_PATHS = (r"(^|/)data/",)

# TorchNEAT ships to PyPI only (no desktop/mobile apps), so no file triggers an "apps" release.
APP_PATHS = ()

# Extra files that repeat the version (none: the version lives in torchneat/_version.py).
VERSION_TARGETS = ()

PYPI_JSON_URL = "https://pypi.org/pypi/{package}/json"

PROJECT_LINKS = {
    "Repository": "https://github.com/KshRaven/torchneat",
    "Release workflow": "https://github.com/KshRaven/torchneat/actions/workflows/release.yml",
    "PyPI": "https://pypi.org/project/torchneat/",
    "TestPyPI": "https://test.pypi.org/project/torchneat/",
    "PyPI publisher settings": "https://pypi.org/manage/project/torchneat/settings/publishing/",
    "TestPyPI publisher settings": "https://test.pypi.org/manage/project/torchneat/settings/publishing/",
}
