# Documentation claim auditor

An offline, deterministic, standard-library-only CLI for checking whether prose
about dividing a picture into labelled regions or masks agrees with Python code.
Python 3.9 or newer is required. No network, LLM, telemetry, or output file writes.

## Installation and usage

```sh
python3 -m pip install .
image-segmentation-claim-audit --version
image-segmentation-claim-audit audit /path/to/repo
python3 -m image_segmentation_claim_audit audit --ignore vendor/ /path/to/repo --json
image-segmentation-claim-audit audit . --ignore build/ --ignore dist/ --fail-on medium --verbose
```

The repository defaults to the current directory. Repeat the single-value ignore
option for each relative path prefix to skip. JSON output is an array of findings;
each contains rule, severity, verdict, and message. Verbose decisions and scan
scope go to standard error. Exit 0 means no findings meet the threshold, 2 means
findings meet it, and 1 means a usage or IO error. The default threshold is high.
Test-only claims have medium severity and fail only at the medium threshold.

```text
documented-image-segmentation-claim-false  HIGH (or MEDIUM for tests only)
documented-no-image-segmentation-claim-false  HIGH
```

## Scan and evidence

Documentation scope is root README.md and Markdown files recursively under docs/.
Triple-backtick fenced blocks are excluded; inline backticks are removed while
their content counts as prose. Matching uses case-insensitive exact substrings,
without stemming, fuzzy matching, or hyphen normalization. Denial spans are removed
first. Longest-first matching blanks spans so each occurrence matches exactly
once under its longest phrase. Repeated occurrences collapse into one finding
per rule family. Denials without contradictory code pass; claims with supporting
code pass. With no phrases, the result is a pass.

Every Python source is checked, including tests, except Git metadata, bytecode
caches, virtual environments, and ignored prefixes. A route must occur in one
module. Imports and aliases are resolved using the AST; local function or class
definitions shadow matching route names. Invalid Python is skipped, with a
verbose diagnostic. No optional model libraries need to be installed.

Eight evidence routes cover model construction paired with inference, two
library function-call families, and generic calls:

```text
1. torchvision.models.segmentation: fcn_resnet50, fcn_resnet101,
   deeplabv3_resnet50, deeplabv3_resnet101, deeplabv3_mobilenet_v3_large,
   lraspp_mobilenet_v3_large; torchvision.models.detection: maskrcnn_resnet50_fpn.
   Imported construction plus direct instance call.
2. ultralytics.YOLO: literal model argument containing "seg", ignoring case,
   plus direct instance call or .predict; not .train/.val/.export/.track alone.
3. transformers.pipeline: exact literal task "image-segmentation" plus instance call.
4. detectron2.engine.DefaultPredictor: module string containing mask_rcnn,
   panoptic, or semantic, ignoring case, plus construction and instance call.
5. segment_anything.SamPredictor (also segment_anything.predictor):
   construction plus .predict.
6. cv2.grabCut or cv2.watershed calls.
7. skimage.segmentation: slic, felzenszwalb, watershed, quickshift,
   chan_vese, morphological_chan_vese, find_boundaries calls.
8. Generic attribute calls on any base, or imported/aliased bare calls:
   segment_image, segment_images, segment_photo, segment_photos,
   segment_picture, segment_pictures, segment_objects, segment_object,
   semantic_segmentation, instance_segmentation, panoptic_segmentation,
   segment_anything, run_segmentation, apply_segmentation, perform_segmentation,
   create_segmentation, generate_segmentation, segment_foreground,
   segment_background, get_segmentation_mask, predict_segmentation,
   segment_regions, segment_pixels, image_segmentation.
```

Bare imports, construction without inference, unbound names, comments, strings,
subprocess commands, and routes split between modules do not suffice. Finding
boxes, cutting out a subject, rectangular cropping, resizing, rotation, flipping,
filling holes, upscaling, adding color, naming classes, captioning, and generation
alone do not suffice. Neither do thresholding, contour extraction, or clustering.

## Differentiation

This tool checks dividing a picture into labelled regions or masks. Neighbours
check these distinct operations; their names are listed separately below.

```text
image-object-detection-claim-audit: finding and boxing things in one picture
image-background-removal-claim-audit: cutting a subject out of a picture
image-cropping-claim-audit: cutting a rectangular region out of a picture
image-resizing-claim-audit: changing a picture's pixel dimensions
image-inpainting-claim-audit: filling a hole in one picture
image-colorization-claim-audit: adding color to a grey picture
super-resolution-claim-audit: model-based upscaling of one picture
image-classification-claim-audit: naming what a picture shows
segmentation-claim-audit: mediapipe-based segmentation only, rather than
  the model and library routes checked here
```

## Limitations

This is static pattern matching, not semantic judgement or runtime validation.
Exact phrases cannot recognize arbitrary restatements. The following exact
substrings also match unrelated prose: the first can describe splitting an image
dataset into training and test parts; the second can describe 3D scene-graph
modelling. These ambiguities are intentional, tested limitations.

```text
segment images
we segment images into training and test sets
segment objects
we segment objects in the scene graph
```

A watershed call on a non-picture array still counts because the call form is
shared. A predictor can count even though its required image initialization is
absent and it would fail at runtime. A non-literal model path never counts, even
when it names suitable weights at runtime. A generic method on an unrelated
object counts because the method name is shared. Relevant spellings:

```text
cv2.watershed
SamPredictor.predict (without requiring .set_image first)
YOLO(variable_path)  # rejected even if the runtime path names a -seg weight file
unrelated.segment_image(...)  # counts through the generic route
```

## Development

```sh
python3 -m unittest discover
python3 -m build
```

The full project audit reports a medium test-only finding from the fixed phrase
oracles. A README-only audit passes without findings. The distribution includes
the discoverable test suite. MIT licensed.
