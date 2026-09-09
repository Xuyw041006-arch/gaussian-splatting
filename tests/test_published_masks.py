import importlib.util
import io
import tempfile
import unittest
import zipfile
from pathlib import Path

from scripts.evaluate_published_masks import evaluate


@unittest.skipUnless(all(importlib.util.find_spec(name) for name in ("numpy", "PIL", "cv2")), "image dependencies unavailable")
class PublishedMasksTests(unittest.TestCase):
    def test_exact_annotation_subset_and_native_metrics(self):
        import numpy as np
        from PIL import Image
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            gt = root / "gt" / "0"
            gt.mkdir(parents=True)
            mask = np.zeros((30, 40), dtype=np.uint8)
            mask[3:22, 4:26] = 255
            image = Image.fromarray(mask)
            image.save(gt / "egg.png")
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            archive = root / "masks.zip"
            with zipfile.ZipFile(archive, "w") as output:
                output.writestr("lerf_mask/ramen/0/egg.png", buffer.getvalue())
                output.writestr("lerf_mask/ramen/1/egg.png", buffer.getvalue())
            result = evaluate(archive, root / "gt", expected_sha256=None)
            self.assertEqual(result["protocol"]["splits"], ["0"])
            self.assertEqual(result["mean_iou"], 1.0)
            self.assertEqual(result["mean_boundary_iou"], 1.0)
            with self.assertRaisesRegex(ValueError, "checksum"):
                evaluate(archive, root / "gt")


if __name__ == "__main__":
    unittest.main()
