"""Check ruler_pred.model_identity on file layouts only (no model load; Codex A3): single-file checkpoint ok; sharded
checkpoint with index + all shards ok and lists every shard; sharded with a missing shard refused; neither file
refused. Exit 1 on failure."""
import json
import os
import shutil
import sys
import tempfile

ROOT = __file__.rsplit("/theory/", 1)[0]
sys.path.insert(0, ROOT + "/src")
import ruler_pred  # noqa: E402

SRC = "/home/kooki3/models/qwen2.5-1.5b-instruct"
bad = []


def base():
    d = tempfile.mkdtemp()
    for f in ("config.json", "tokenizer.json", "tokenizer_config.json", "generation_config.json"):
        shutil.copy(os.path.join(SRC, f), d)
    return d


d = base(); open(os.path.join(d, "model.safetensors"), "wb").write(b"x")
if "model:model.safetensors" not in ruler_pred.model_identity(d, "abc"):
    bad.append("single-file layout")
d = base()
json.dump({"weight_map": {"a": "model-00001-of-00002.safetensors", "b": "model-00002-of-00002.safetensors"}},
          open(os.path.join(d, "model.safetensors.index.json"), "w"))
for f in ("model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"):
    open(os.path.join(d, f), "wb").write(b"y")
ident = ruler_pred.model_identity(d, "rev1")
if not all(f"model:model-0000{i}-of-00002.safetensors" in ident for i in (1, 2)) or ident["model:revision"] != "rev1":
    bad.append(f"sharded layout {sorted(ident)}")
os.remove(os.path.join(d, "model-00002-of-00002.safetensors"))
try:
    ruler_pred.model_identity(d)
    bad.append("missing shard accepted")
except FileNotFoundError:
    pass
d = base()
try:
    ruler_pred.model_identity(d)
    bad.append("no weights accepted")
except SystemExit:
    pass
print("PROBLEMS", bad if bad else "none")
sys.exit(1 if bad else 0)
