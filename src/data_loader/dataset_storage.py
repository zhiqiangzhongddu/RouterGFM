"""Utilities for accessing and modifying the underlying data storage of PyG datasets, including support for unwrapping Subset datasets to access their base data storage."""

def _get_dataset_data_storage(dataset):
    """Return the underlying PyG Data storage for datasets that expose one."""
    data_obj = getattr(dataset, "data", None)
    if data_obj is not None:
        return data_obj
    inner = getattr(dataset, "_data", None)
    if inner is not None:
        return inner
    return None


def _set_dataset_data_storage(dataset, data_obj) -> None:
    """Best-effort update of a dataset's in-memory Data storage."""
    if hasattr(dataset, "_data"):
        try:
            dataset._data = data_obj
            return
        except Exception:
            pass
    if hasattr(dataset, "data"):
        try:
            dataset.data = data_obj
        except Exception:
            pass


def _unwrap_subset_dataset(dataset):
    current = dataset
    while hasattr(current, "dataset") and hasattr(current, "indices"):
        parent = getattr(current, "dataset", None)
        if parent is None or parent is current:
            break
        current = parent
    return current
