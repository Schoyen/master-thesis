import os

import pandas as pd

df = pd.read_pickle(os.path.join("dat", "gs_df.pkl"))

df.to_csv(
    os.path.join("dat", "gs_df.csv"),
    sep="&",
    na_rep="x",
    line_terminator=r"\\" + "\n",
    index=False,
    float_format="%.4f",
)
