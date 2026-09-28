"""Fill kiui/typing.py when a release ships that module empty.

kiui 0.3.5 imports Union, Tensor, and ndarray from kiui.typing, but the
published module is only a comment. train.py imports kiui.cam at startup,
so the process dies before argument parsing.
"""

from pathlib import Path

STUB = """\
from typing import *

import numpy as np
import torch

Tensor = torch.Tensor
ndarray = np.ndarray
"""


def main() -> None:
    import kiui

    path = Path(kiui.__file__).resolve().parent / "typing.py"
    text = path.read_text()
    if "from typing import *" in text and "Tensor = torch.Tensor" in text:
        print(f"kiui typing already usable: {path}")
        return
    path.write_text(STUB)
    print(f"wrote kiui typing stub: {path}")


if __name__ == "__main__":
    main()
