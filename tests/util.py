# util.py
# shared test scaffolding. TempDirsMixin repoints every config path at a
# throwaway directory so tests can never touch the real catalog, real
# thumbnails, real exports, or the real footage. a scan or analyze
# without --root lands in the empty self.tmp / "media".

import tempfile
from pathlib import Path

from clipengine import config


class TempDirsMixin:
    def setUp(self):
        super().setUp()
        self._tmpdir = tempfile.TemporaryDirectory(prefix="clipengine-test-")
        # a cleanup, not tearDown, so a test's own later cleanups (a chdir
        # back out) run before the folder goes
        self.addCleanup(self._tmpdir.cleanup)
        self.tmp = Path(self._tmpdir.name)
        self._saved = (config.DATA_DIR, config.DB_PATH, config.THUMB_DIR,
                       config.EXPORT_DIR, config.MEDIA_ROOT)
        config.DATA_DIR = self.tmp / "data"
        config.DB_PATH = config.DATA_DIR / "catalog.db"
        config.THUMB_DIR = config.DATA_DIR / "thumbs"
        config.EXPORT_DIR = self.tmp / "exports"
        config.MEDIA_ROOT = self.tmp / "media"
        config.MEDIA_ROOT.mkdir()
        config.create_directories()

    def tearDown(self):
        (config.DATA_DIR, config.DB_PATH, config.THUMB_DIR,
         config.EXPORT_DIR, config.MEDIA_ROOT) = self._saved
        super().tearDown()
