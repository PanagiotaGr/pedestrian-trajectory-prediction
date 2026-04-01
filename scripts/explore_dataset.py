import os

root = os.path.expanduser("~/imptc_project/data/imptc_trajectory_dataset/train")

scenes = sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))])

print(f"Number of scenes: {len(scenes)}")
print("First 10 scenes:", scenes[:10])

sample_scene = scenes[0]
sample_path = os.path.join(root, sample_scene)

print(f"\nFiles in scene {sample_scene}:")
for f in os.listdir(sample_path):
    print(" -", f)
