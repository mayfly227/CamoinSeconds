import csv
import argparse
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import DataLoader, Dataset


SUPPORTED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

CATEGORY_COLUMN_CANDIDATES = ["class_name", "category", "class", "class_id"]
PROMPT_COLUMN_CANDIDATES = ["prompt", "text", "caption"]
IMAGE_PATH_COLUMN_CANDIDATES = ["image_path", "image", "path", "file_name"]


@dataclass(frozen=True)
class PromptRecord:
	category: str
	prompt: str
	image_path: str


class PromptImageMapDataset(Dataset):
	def __init__(
		self
	):
		prompt_map_file = '/home/dk/project/FastAdv/data/generated_by_prompt/prompt_image_map_all.txt'
		with open(prompt_map_file, "r", encoding="utf-8") as f:
			



	def get_category_counts(self) -> Dict[str, int]:
		counts: Dict[str, int] = defaultdict(int)
		for record in self._active_records:
			counts[record.category] += 1
		return dict(sorted(counts.items(), key=lambda item: item[0]))

	def __len__(self):
		return len(self._active_records)

	def __getitem__(self, idx: int):
		record = self._active_records[idx]
		return {
			"prompt": record.prompt,
			"category": record.category,
			"image_path": record.image_path,
		}


def prompt_collate_fn(examples: Sequence[dict]) -> dict:
	return {
		"prompts": [example["prompt"] for example in examples],
		"categories": [example["category"] for example in examples],
		"image_paths": [example["image_path"] for example in examples],
	}


def build_prompt_dataloader(
	prompt_map_file: str,
	batch_size: int,
	selected_categories: Optional[Iterable[str]] = None,
	samples_per_category: Optional[int] = None,
	seed: Optional[int] = None,
	resample_mode: str = "once",
	drop_missing_images: bool = True,
	category_column: Optional[str] = None,
	prompt_column: Optional[str] = None,
	image_path_column: Optional[str] = None,
	data_root: Optional[str] = None,
	shuffle: bool = True,
	num_workers: int = 0,
) -> Tuple[DataLoader, PromptImageMapDataset]:

	dataset = PromptImageMapDataset(
		records=records,
		samples_per_category=samples_per_category,
		seed=seed,
		resample_mode=resample_mode,
	)
	dataloader = DataLoader(
		dataset,
		batch_size=batch_size,
		shuffle=shuffle,
		num_workers=num_workers,
		collate_fn=prompt_collate_fn,
		pin_memory=torch.cuda.is_available(),
	)
	return dataloader, dataset


def _run_main_test():
	parser = argparse.ArgumentParser(description="Test prompt dataset and dataloader.")
	parser.add_argument(
		"--prompt_map_file",
		type=str,
		default="data/generated_by_prompt/prompt_image_map_all.txt",
		help="Path to metadata map file.",
	)
	parser.add_argument(
		"--selected_categories",
		nargs="*",
		default=None,
		help="Selected categories, supports multi-value and comma-separated input.",
	)
	parser.add_argument(
		"--samples_per_category",
		type=int,
		default=None,
		help="Max sampled records per category.",
	)
	parser.add_argument(
		"--resample_mode",
		type=str,
		choices=["once", "epoch"],
		default="once",
		help="Sampling mode.",
	)
	parser.add_argument("--batch_size", type=int, default=1)
	parser.add_argument("--seed", type=int, default=123)
	parser.add_argument("--drop_missing_images", action="store_true")
	parser.add_argument("--data_root", type=str, default=None)
	parser.add_argument("--num_preview", type=int, default=5)
	args = parser.parse_args()

	loader, dataset = build_prompt_dataloader(
		prompt_map_file=args.prompt_map_file,
		batch_size=args.batch_size,
		selected_categories=args.selected_categories,
		samples_per_category=args.samples_per_category,
		seed=args.seed,
		resample_mode=args.resample_mode,
		drop_missing_images=args.drop_missing_images,
		data_root=args.data_root,
		shuffle=True,
		num_workers=0,
	)

	print("=== Dataset Summary ===")
	print(f"map_file: {args.prompt_map_file}")
	print(f"dataset_size: {len(dataset)}")
	print(f"category_counts: {dataset.get_category_counts()}")

	preview_count = min(max(args.num_preview, 0), len(dataset))
	print(f"\n=== First {preview_count} Samples ===")
	for i in range(preview_count):
		sample = dataset[i]
		print(
			f"[{i}] category={sample['category']} | prompt={sample['prompt'][:80]} | image={Path(sample['image_path']).name}"
		)

	batch = next(iter(loader))
	print("\n=== One Batch ===")
	print(f"batch_size: {len(batch['prompts'])}")
	print(f"batch_categories: {batch['categories']}")
	if batch["image_paths"]:
		first_image = Path(batch["image_paths"][0])
		print(f"first_image_path: {first_image}")
		print(f"first_image_exists: {first_image.exists()}")


if __name__ == "__main__":
	_run_main_test()
