# Datasets API

:material-file-code: `datasets/`

---

## Dataset Registry

:material-file-code: `datasets/__init__.py`

### `registerDataset(name)`

Decorator that registers a dataset class in the global `DATASET_REGISTRY`.

```python
@registerDataset('celebahq')
class CelebAHQ(Dataset):
    ...
```

| Parameter | Type | Description |
|:----------|:-----|:------------|
| `name` | `str` | Registry key (e.g. `"celebahq"`) |

---

### `getDataset(name) → type`

Retrieve a registered dataset class by name.

| Parameter | Type | Description |
|:----------|:-----|:------------|
| `name` | `str` | Registered dataset name |

**Raises:** `KeyError` if the name is not found.

**Usage:**

```python
from datasets import getDataset

DatasetClass = getDataset("celebahq")
dataset = DatasetClass(rootPath, download=True, transforms=transforms)
```

---

## CelebAHQ

:material-file-code: `datasets/celebahq.py`

Wrapper for the CelebA-HQ dataset. Handles downloading from Google Drive, extraction, and loading.

```python
@registerDataset('celebahq')
class CelebAHQ(Dataset):
    def __init__(self, rootPath, download=False, transforms=None)
```

| Parameter | Type | Default | Description |
|:----------|:-----|:--------|:------------|
| `rootPath` | `pathlib.Path` | — | Root directory for the dataset |
| `download` | `bool` | `False` | Download from Google Drive if not present |
| `transforms` | `torchvision.transforms.Compose` | `None` | Transforms applied to each image |

### Dataset Details

| Property | Value |
|:---------|:------|
| **Images** | ~30,000 face images |
| **Resolution** | 256×256 |
| **Source** | Google Drive (via `gdown`) |
| **Format** | JPEG/PNG individual files |
| **Storage Path** | `{rootPath}/CelebA-HQ/` |

### Methods

#### `__len__() → int`

Returns the number of images in the dataset.

#### `__getitem__(idx) → Tensor`

Returns the image at index `idx`, converted to RGB. If transforms are provided, they are applied.

| Output | Type | Shape (with default transforms) |
|:-------|:-----|:------|
| image | `Tensor` | `[3, 256, 256]` in range `[-1, 1]` |

#### `printDatasetInfo()`

Prints dataset metadata (root path, zip file path, URL) to the console using Rich formatting.

### Download Flow

When `download=True` and the dataset doesn't exist:

1. Downloads `CelebAHQ.zip` from Google Drive using `gdown`
2. Extracts the ZIP archive
3. Renames the extracted `images/` folder to `CelebA-HQ/`
4. Deletes the ZIP file

!!! warning "First download"
    The download is ~1.3 GB and requires a stable internet connection. The Google Drive link may have a daily download quota.

### Recommended Transforms

```python
from torchvision import transforms as T

transforms = T.Compose([
    T.RandomHorizontalFlip(),
    T.ToTensor(),
    T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Maps [0,1] → [-1,1]
])
```
