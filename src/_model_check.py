from pathlib import Path
print(len([d for d in Path("src/roboflow_dataset/train").iterdir() if d.is_dir()]))