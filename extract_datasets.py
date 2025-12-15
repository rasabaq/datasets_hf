import os
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
from huggingface_hub import list_datasets
from huggingface_hub.errors import HfHubHTTPError
from tqdm import tqdm

# Add the parent of HuggingFace (i.e., the actual project root)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

BATCH_SIZE = 50_000  # number of datasets to process per batch
PROGRESS_STEP = 1_000  # refresh tqdm every this many rows
MAX_RETRIES = 5  # transient 5xx retries

EXPAND_FIELDS = [
    "author",
    # "citation",
    "createdAt",
    # "description",
    # "disabled",
    "downloads",
    "downloadsAllTime",
    # "gated",
    # "lastModified",
    "likes",
    # "paperswithcode_id",
    # "sha",
    # "private",
    # "siblings",
    "tags",
]


def stream_dataset_batches(batch_size: int):
    """Yield batches of datasets pulled from the Hub, retrying on transient 5xx errors."""
    seen_ids = set()
    batch = []
    attempt = 0

    while True:
        try:
            for dataset in list_datasets(
                sort="likes",
                direction=-1,
                expand=EXPAND_FIELDS,
            ):
                if dataset.id in seen_ids:
                    continue
                seen_ids.add(dataset.id)
                batch.append(dataset)
                if len(batch) >= batch_size:
                    yield batch
                    batch = []
            break  # completed without error
        except HfHubHTTPError as e:
            status = getattr(getattr(e, "response", None), "status_code", None)
            if status not in {502, 503, 504} or attempt >= MAX_RETRIES:
                raise
            attempt += 1
            wait = min(60, 2 ** attempt)
            print(f"Hub returned {status}; retrying {attempt}/{MAX_RETRIES} after {wait}s...")
            time.sleep(wait)
            continue

    if batch:
        yield batch


def dataset_to_dict(dataset):
    """Convert a HuggingFace dataset object into a dict for the DataFrame."""
    return {
        "id": dataset.id,
        "author": dataset.author,
        # "citation": getattr(dataset, "citation", None),
        "created_at": dataset.created_at,
        # "description": str(getattr(dataset, "description", None)).replace("\n", "").replace("\t", ""),
        # "sha": dataset.sha,
        # "last_modified": dataset.last_modified,
        # "private": dataset.private,
        # "gated": dataset.gated,
        # "disabled": dataset.disabled,
        "downloads": dataset.downloads,
        "downloads_all_time": dataset.downloads_all_time,
        "likes": dataset.likes,
        # "paperswithcode_id": dataset.paperswithcode_id,
        "tags": ",".join(dataset.tags) if dataset.tags else None,
        # "card_data": dataset.card_data,
        # "siblings": dataset.siblings,
    }


def main():
    today = datetime.today().strftime("%Y-%m-%d")
    hf_folder = Path("Datasets")
    hf_folder.mkdir(parents=True, exist_ok=True)
    filename = hf_folder / f"hf_datasets_all_{today}.csv"

    progress_buffer = 0
    write_header = True
    with tqdm(desc="Processing metrics", unit="ds", dynamic_ncols=True) as progress:
        for batch in stream_dataset_batches(BATCH_SIZE):
            records = []
            for ds in batch:
                records.append(dataset_to_dict(ds))
                progress_buffer += 1
                if progress_buffer >= PROGRESS_STEP:
                    progress.update(progress_buffer)
                    progress_buffer = 0

            # flush any remaining progress for this batch
            if progress_buffer:
                progress.update(progress_buffer)
                progress_buffer = 0

            df_batch = pd.DataFrame(records)
            mode = "w" if write_header else "a"
            df_batch.to_csv(filename, index=False, mode=mode, header=write_header)
            write_header = False


if __name__ == "__main__":
    main()
