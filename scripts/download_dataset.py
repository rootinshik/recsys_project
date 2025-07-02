import argparse
from pathlib import Path
from typing import Literal

from datasets import Dataset, DatasetDict, load_dataset

from src.utils.logger import get_logger

logger = get_logger("download_dataset")


class YambdaDataset:
    """https://huggingface.co/datasets/yandex/yambda#download"""

    INTERACTIONS = frozenset(
        ["likes", "listens", "multi_event", "dislikes", "unlikes", "undislikes"]
    )

    def __init__(
        self,
        dataset_type: Literal["flat", "sequential"] = "flat",
        dataset_size: Literal["50m", "500m", "5b"] = "50m",
    ):
        assert dataset_type in {"flat", "sequential"}
        assert dataset_size in {"50m", "500m", "5b"}
        self.dataset_type = dataset_type
        self.dataset_size = dataset_size

    def interaction(
        self,
        event_type: Literal[
            "likes", "listens", "multi_event", "dislikes", "unlikes", "undislikes"
        ],
    ) -> Dataset:
        assert event_type in YambdaDataset.INTERACTIONS
        return self._download(f"{self.dataset_type}/{self.dataset_size}", event_type)

    def audio_embeddings(self) -> Dataset:
        return self._download("", "embeddings")

    def album_item_mapping(self) -> Dataset:
        return self._download("", "album_item_mapping")

    def artist_item_mapping(self) -> Dataset:
        return self._download("", "artist_item_mapping")

    @staticmethod
    def _download(data_dir: str, file: str) -> Dataset:
        data = load_dataset(
            "yandex/yambda", data_dir=data_dir, data_files=f"{file}.parquet"
        )
        assert isinstance(data, DatasetDict)
        return data["train"]


def parse_args():
    parser = argparse.ArgumentParser(description="Скачать Yandex Yambda датасет")

    parser.add_argument(
        "--type",
        choices=["flat", "sequential"],
        default="flat",
        help="Тип датасета (по умолчанию: flat)",
    )
    parser.add_argument(
        "--size",
        choices=["50m", "500m", "5b"],
        default="50m",
        help="Размер датасета (по умолчанию: 50m)",
    )
    parser.add_argument(
        "--embeddings",
        action="store_true",
        help="Скачать дополнительные таблицы c эмбедингами",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    output_dir = Path("data/raw")
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"📦 Загружаем датасет: type={args.type}, size={args.size}")
    dataset = YambdaDataset(dataset_type=args.type, dataset_size=args.size)

    extra_datasets = {
        "album_item_mapping": dataset.album_item_mapping,
        "artist_item_mapping": dataset.artist_item_mapping,
    }

    if args.embeddings:
        extra_datasets["audio_embeddings"] = dataset.audio_embeddings

    for event in YambdaDataset.INTERACTIONS:
        logger.info(f"⬇️ Скачиваем interaction: {event}")
        ds = dataset.interaction(event)
        output_path = output_dir / f"{event}.parquet"
        logger.info(f"💾 Сохраняем в: {output_path}")
        ds.to_parquet(output_path)

    for name, loader_fn in extra_datasets.items():
        logger.info(f"⬇️ Скачиваем: {name}")
        ds = loader_fn()
        output_path = output_dir / f"{name}.parquet"
        logger.info(f"💾 Сохраняем в: {output_path}")
        ds.to_parquet(output_path)

    logger.info("✅ Загрузка завершена.")


if __name__ == "__main__":
    main()
