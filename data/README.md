# Data

This project uses the **PhiUSIIL Phishing URL Dataset** from the UCI Machine Learning Repository.

The raw dataset is not included in this repository.

The experiment script downloads the dataset automatically using:

```python
from ucimlrepo import fetch_ucirepo
dataset = fetch_ucirepo(id=967)
```

This keeps the repository lightweight and avoids uploading large dataset files directly to GitHub.
