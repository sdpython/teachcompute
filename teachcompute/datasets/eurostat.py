import os
import gzip
import urllib.request
from typing import Optional
import numpy
import pandas
from .. import log


def mortality_table(
    to: str = ".", stop_at: Optional[int] = None, verbose: bool = False
) -> pandas.DataFrame:
    """
    This function retrieves mortality table from EuroStat or INSEE.
    A copy is provided. The link is changing.

    :param to: data needs to be downloaded, location of this place
    :param stop_at: the overall process is quite long, if not None,
        it only keeps the first rows
    :return: data_frame

    The function checks the file final_name exists.
    If it is the case, the data is not downloaded twice.
    The header contains a weird format as coordinates are separated by a comma::

        indic_de,sex,age,geo\\time	2013 	2012 	2011 	2010 	2009

    We need to preprocess the data to split this information into columns.
    The overall process takes 4-5 minutes, 10 seconds to download (< 10 Mb),
    4-5 minutes to preprocess the data (it could be improved). The processed data
    contains the following columns::

        ['annee', 'valeur', 'age', 'age_num', 'indicateur', 'genre', 'pays']

    Columns *age* and *age_num* look alike. *age_num* is numeric and is equal
    to *age* except when *age_num* is 85. Everybody above that age
    fall into the same category. The table contains many indicators:

    * PROBSURV: Probabilité de survie entre deux âges exacts (px)
    * LIFEXP: Esperance de vie à l'âge exact (ex)
    * SURVIVORS: Nombre des survivants à l'âge exact (lx)
    * PYLIVED: Nombre d'années personnes vécues entre deux âges exacts (Lx)
    * DEATHRATE: Taux de mortalité à l'âge x (Mx)
    * PROBDEATH: Probabilité de décès entre deux âges exacts (qx)
    * TOTPYLIVED: Nombre total d'années personne vécues après l'âge exact (Tx)
    """

    final_name = os.path.join(to, "mortality.txt")
    if os.path.exists(final_name) and os.stat(final_name).st_size > 1e7:
        return final_name

    dest = os.path.join(to, "demo_mlifetable.tsv.gz")
    if not os.path.exists(dest) or os.stat(dest).st_size < 1e7:
        url = (
            "https://github.com/sdpython/data/raw/main/mortality/demo_mlifetable.tsv.gz"
        )
        with urllib.request.urlopen(url) as u:
            content = u.read()
        with open(dest, "wb") as f:
            f.write(content)
        with gzip.open(dest, "rb") as fg:
            file_content = fg.read()
        content = str(file_content, encoding="utf8")
        with open(final_name, "w", encoding="utf8") as ft:
            ft.write(content)

    def format_age(s):
        "local function"
        if s.startswith("Y_"):
            if s.startswith("Y_LT"):
                return "YLT" + s[4:]
            if s.startswith("Y_GE"):
                return "YGE" + s[4:]
            raise SyntaxError(s)  # pragma: no cover
        i = int(s.strip("Y"))
        return "Y%02d" % i

    def format_age_num(s):
        "local function"
        if s.startswith("Y_"):
            if s.startswith("Y_LT"):
                return float(s.replace("Y_LT", ""))
            if s.startswith("Y_GE"):
                return float(s.replace("Y_GE", ""))
            raise SyntaxError(s)  # pragma: no cover
        i = int(s.strip("Y"))
        return float(i)

    def format_value(s):
        "local function"
        if s.strip() == ":":
            return numpy.nan
        return float(s.strip(" ebp"))

    log(verbose, lambda: "[mortality_table] read")
    dff = pandas.read_csv(final_name, sep="\t", encoding="utf8")

    if stop_at is not None:
        if verbose:
            print(f"[mortality_table] read only {stop_at} rows")
        dfsmall = dff.head(n=stop_at)
        df = dfsmall
    else:
        df = dff

    log(verbose, lambda: f"[mortality] step 1, shape is {df.shape}")
    dfi = df.reset_index().set_index("indic_de,sex,age,geo\\time")
    dfi = dfi.drop("index", axis=1)
    dfs = dfi.stack()
    dfs = pandas.DataFrame({"valeur": dfs})

    log(verbose, lambda: f"[mortality] step 2, shape is {dfs.shape}")
    dfs["valeur"] = dfs["valeur"].astype(str)
    dfs["valeur"] = dfs["valeur"].apply(format_value)
    dfs = dfs[dfs.valeur >= 0].copy()
    dfs = dfs.reset_index()
    dfs.columns = ["index", "annee", "valeur"]

    log(verbose, lambda: f"[mortality] step 3, shape is {dfs.shape}")
    dfs["age"] = dfs["index"].apply(lambda i: format_age(i.split(",")[2]))
    dfs["age_num"] = dfs["index"].apply(lambda i: format_age_num(i.split(",")[2]))
    dfs["indicateur"] = dfs["index"].apply(lambda i: i.split(",")[0])
    dfs["genre"] = dfs["index"].apply(lambda i: i.split(",")[1])
    dfs["pays"] = dfs["index"].apply(lambda i: i.split(",")[3])

    log(verbose, lambda: f"[mortality] step 4, shape is {dfs.shape}")
    dfy = dfs.drop("index", axis=1)
    dfy.to_csv(final_name, sep="\t", encoding="utf8", index=False)
    return final_name
