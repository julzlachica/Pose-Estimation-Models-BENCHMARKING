import json, sys
from collections import Counter

d = json.load(open(sys.argv[1], encoding="utf-8"))
print("images:", len(d["images"]), "| annotations:", len(d["annotations"]))

lens = Counter(len(a.get("keypoints", [])) for a in d["annotations"])
print("keypoint array lengths:", dict(lens))

for c in d.get("categories", []):
    if "keypoints" in c:
        print("skeleton:", c["name"], len(c["keypoints"]), "points ->", c["keypoints"])

per_img = Counter(a["image_id"] for a in d["annotations"])
print("images with 1 person:", sum(v == 1 for v in per_img.values()),
      "| with 2+:", sum(v > 1 for v in per_img.values()))
