# Dataset provenance and curation

Domain: natural photographs of dogs. Breed variation, fur appearance, framing,
background and illumination make a useful focused domain. A single animal category
is more tractable at 64x64 than unrestricted scenes, while 120 breeds offer richer
conditioning than a one-class corpus. The starting source is an existing collection;
this project contributes automated filtering, deduplication, preprocessing,
annotation-derived captions, independent generative splits and auditable manifests.
It does not claim to have photographed or originally assembled the source images.

Primary dataset: Aditya Khosla, Nityananda Jayadevaprakash, Bangpeng Yao and Li
Fei-Fei. Novel Dataset for Fine-Grained Image Categorization. FGVC/CVPR Workshop,
2011. http://vision.stanford.edu/aditya86/ImageNetDogs/
Paper: https://people.csail.mit.edu/khosla/papers/fgvc2011.pdf
Underlying ImageNet: Deng et al., ImageNet: A Large-Scale Hierarchical Image
Database, CVPR 2009. https://www.image-net.org/

Acquisition mirror: https://huggingface.co/datasets/Alanox/stanford-dogs
Pinned revision: `00c2ba455d1c85a01f14c6c5e1ec8524eeb33a1f`.
Archive SHA256:
`572e8ab5097aac3f7cbf18fd7d40b752169ba188e89b12b5b9bf682157d52333`.
Both archive and metadata hashes are enforced by the downloader. Metadata associates
source filenames with breed names and annotated boxes. The original server could
not be reached during development. The mirror image archive was successfully
retrieved and hash-verified. It stores flattened image filenames; provenance
retains these filenames and the mirror member path.

Usage: ImageNet/Stanford Dogs images have original photographer copyrights.
Public research access does not transfer ownership. Use for this non-commercial
educational assignment under source terms; seek separate permission for commercial
use or public redistribution. The mirror's MIT tag is not adopted as the photo
license. Individual photo URLs, photographers and licenses are not supplied by this
mirror; fields record that limitation rather than invent attribution. The source
member identifier supports lookup against the original collection. Distribute code
and the manifest separately from restricted original images where necessary.

Manifest schema (JSONL, paths relative to manifest):

| Fields | Meaning |
|---|---|
| id | First 24 hex characters of original file SHA256 |
| source, acquisition_url, source_revision | Primary provenance and exact mirror object |
| source_member, source_image_id | Archive member and original image identifier |
| metadata_url, metadata_sha256 | Annotation provenance |
| image_path | Curated RGB PNG used without further manual conversion |
| original_width, original_height | Decoded EXIF-corrected source dimensions |
| width, height | Curated dimensions, both 64 |
| caption, caption_method | Actual derived description and algorithm version |
| breed, bounding_boxes | Source annotations |
| scene_colors, scene_brightness | Measured scene appearance, not fur claims |
| license | Known usage restriction and retained rights |
| content_hash | Original file SHA256 |
| pixel_hash, processed_hash | Curated decoded pixels and PNG-file SHA256 |
| phash | Original visual 63-bit DCT fingerprint |
| split | train, validation or test |

The largest annotated dog determines framing and horizontal position. All boxes
are retained. Palette nearest-centroid counts over RGB pixels supply two dominant
scene colors, and mean RGB supplies brightness bins. They are coarse descriptions,
not semantic background recognition. Source annotation errors and distractors can
remain; inspect a random audit sample before using this as a final dataset.

Perceptual deduplication rejects a new image if within Hamming radius 4 of any
accepted original-image fingerprint. Sorted member order makes retention stable.
This is conservative but not invariant to all crops, overlays or mirror flips.
Exact/processed SHA256 and perceptual checks are complementary, not guarantees of
world-wide photograph uniqueness. The independent verifier checks every retained
file and accepted pair against these implemented rules.
