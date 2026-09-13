from __future__ import annotations

import base64
import zlib

import pytest

from dfn_cave_studio.services.import_service import read_input_table


_SYNTHETIC_XLS = (
    "eNrtWE1oE0EU/mbz0yS2TVJToRVqKFi1tmDw4qVNFbQXDdUcrKLErVnMkpiEuAr2YrXmKAieLF4qBfFS9eIPKujNg1DRgyAIiR49"
    "CQoe2qxvXjYh1R4a0OLPfmG+ee/NzL6X+d99tRAszd7rLuMHDMOBiumFu8EmKHlrSgBUbppSrOUeSqaNvwpeDw2k24XHbS9b5Bj"
    "K8S5DwV3nc2LgA6VjyCOWy2rhNcQejkEVMoYhYoEbZGlHF0fVwXySeT3zHa75hHmELVeYh6huSRzFQjTWv8uaxeNKL5e1Qz73A"
    "bd5x5YIOvFCzuILV0W1rgu7C7qa+TMLepytmAON26iW1QpqpoQQDeAcvpph4EttpT4L2/a1tQuQ/dtye8sK9muKE5iCeYIneFF"
    "OSGd1EcZTmmZEFnGcN1WZaJRTuYyW0JMkaeoZQ8+eIns2VzBSJNK+rGW0c6qh57LrACNnqJlEUssbKaquTuqnzxopB5DU81Qzfi"
    "Q2OL4/PhjxyR2dd4DAsh2gjVdGK3ESfpaDvD4CFMri7c+vD0yMRRNsmeLgqmfDZo7exEXZghq3c4nCLB/dz/J25kv81I0sdzOHa"
    "FZT3jfWaQn7prnOZS7tIz87GW+iWxrkrSQXPx182FP8GN1G8vxoeTI0/zY6i146q5LUXv6mMSAGxMx1iUfRWi6sfeQ9c9dPe4pH"
    "CVixm9YB6McSfCwGmaua7B1R1xSrr6qagzRHXXOS5qxrLtJcdc1NmtvyKlbwKtir7Men1JOCvaYD0nZrk2T/Dsk3uZbCPEMHsxvD"
    "Sgfu8yCMNJzlPtiwYcOGDRs2bNj4HyGsG7qDb6fg+6jbuq/L7zpLlCr2Z5J/FoeQo59BL6Z7kaW8gPNNzZ8NcInas8Qq29S+F0"
    "ocJu8FpDHBcaSbnr/0Bica/8+qGwZ+3RJq1n+lmTh/s//vqcbftw=="
)


def test_empty_unnamed_csv_column_is_ignored(tmp_path) -> None:
    path = tmp_path / "synthetic.csv"
    path.write_text("hole_id,from_depth,to_depth,rmr,\nSYN-1,0,1,0,\n", encoding="utf-8")
    frame = read_input_table(str(path))
    assert list(frame.columns) == ["hole_id", "from_depth", "to_depth", "rmr"]
    assert frame.iloc[0]["rmr"] == 0


def test_nonempty_unnamed_csv_column_is_rejected(tmp_path) -> None:
    path = tmp_path / "synthetic.csv"
    path.write_text("hole_id,from_depth,to_depth,rmr,\nSYN-1,0,1,50,unexpected\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Unnamed input column"):
        read_input_table(str(path))


def test_empty_unnamed_xlsx_column_is_ignored(tmp_path) -> None:
    path = tmp_path / "synthetic.xlsx"
    import pandas as pd

    pd.DataFrame(
        {"hole_id": ["SYN-1"], "from_depth": [0], "to_depth": [1], "rqd": [100], "Unnamed: 4": [None]}
    ).to_excel(path, index=False)
    assert list(read_input_table(str(path)).columns) == ["hole_id", "from_depth", "to_depth", "rqd"]


def test_legacy_xls_is_read_with_ci_runtime_dependency(tmp_path) -> None:
    """Exercise the real xlrd/BIFF path with a fully synthetic workbook."""
    path = tmp_path / "synthetic-collars.xls"
    path.write_bytes(zlib.decompress(base64.b64decode(_SYNTHETIC_XLS)))

    frame = read_input_table(str(path))

    assert list(frame.columns) == [
        "hole_id",
        "easting",
        "northing",
        "elevation",
        "total_depth",
        "azimuth",
        "dip",
    ]
    assert frame.iloc[0].to_dict() == {
        "hole_id": "SYN-XLS-1",
        "easting": 10.5,
        "northing": 20.25,
        "elevation": 30.75,
        "total_depth": 40,
        "azimuth": 0,
        "dip": -90,
    }
