# deepstream-parsers

Custom NVIDIA DeepStream `nvinfer` output parsers that read **stock model
exports** directly.

## Why custom parsers are needed at all

`nvinfer` runs a model but cannot interpret its raw output tensor. A parser
translates that tensor into `NvDsObjectMeta` on the frame, which is what every
native downstream element consumes — `nvtracker`, `nvdsosd`, `nvdsanalytics`,
`nvmsgconv`. Without one, the results exist only as a raw tensor and nothing
downstream can see them.

## Why not just use the vendor parsers

The widely used vendor parsers expect models re-exported through their own
scripts. Those scripts do not merely convert format — they reshape the graph
before the ONNX is written, typically transposing channel-first output to
row-major, taking an argmax over class scores, and converting boxes from
centre-form to corner-form.

That works, but it produces a **DeepStream-specific model file** that other
runtimes cannot consume. These parsers take the opposite approach: keep the
stock export so every runtime reads one model file, and absorb the layout
difference in C++ instead.

## Design notes

Common to the parsers here:

- **Channel-first input.** Stock exports are typically `[1, C, N]` — all N values
  of one channel, then the next. Indexing is `buf[c * N + n]`, not `buf[n * C + c]`.
- **Boxes converted in the parser.** Stock exports usually emit centre-form
  (`cx, cy, w, h`); `NvDsInferParseObjectInfo` wants corner-form.
- **Class chosen in the parser.** Stock exports keep per-class scores separate
  rather than pre-argmaxed.

## Parsers

| Directory | Exported symbol | Reads |
|---|---|---|
| `lpr_bbox/` | `NvDsInferParseCustomYoloLPR` | single-class YOLO detector, `[1, 5, N]` channel-first (`cx, cy, w, h, conf`) |
| `classifier_softmax/` | `NvDsInferClassiferParseSoftmaxTop1` | full-frame classifier, raw class scores (logits); reports the best class with its softmax probability (`parse-classifier-func-name`, `network-type=1`) |

### Why a classifier needs one

`nvinfer` decodes a classifier itself, but its built-in parser reports the raw maximum
score, a logit for a stock export without a Softmax layer, and takes names from a
DeepStream-format label file (`;`-separated per output layer). With an ordinary
one-label-per-line file the class has no label and nothing is attached. The classifier
parser applies softmax and always labels the class with its id, so a probability is
attached on every frame.

## Building

Only source is tracked. A built `.so` is specific to one JetPack / DeepStream /
CUDA / architecture combination, so each machine builds its own. Each directory
is standalone:

```bash
export CUDA_VER=11.4                       # match the target's CUDA
make -C lpr_bbox CUDA_VER=$CUDA_VER
```

## Verifying a build

Check this before wiring a parser into an `nvinfer` config — the exported symbol
is what `parse-bbox-func-name` must name, and a mismatch surfaces only at
runtime, as zero detections rather than an error:

```bash
nm -D --defined-only lpr_bbox/libnvdsinfer_custom_impl_Yolo_LPR.so | grep NvDsInferParse
ldd lpr_bbox/libnvdsinfer_custom_impl_Yolo_LPR.so
```

## Using one

```
[property]
...
parse-bbox-func-name=NvDsInferParseCustomYoloLPR
custom-lib-path=/abs/path/to/lpr_bbox/libnvdsinfer_custom_impl_Yolo_LPR.so
cluster-mode=4
```

Note that `output-tensor-meta=1` and a custom bbox parser are effectively
mutually exclusive — with both enabled, some DeepStream versions skip bbox
parsing entirely and yield no detections.
