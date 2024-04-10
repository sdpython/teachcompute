# coding: utf-8
import os
import sys

from sphinx_runpython.conf_helper import has_dvipng, has_dvisvgm
from sphinx_runpython.github_link import make_linkcode_resolve

from teachcompute import __version__

extensions = [
    "nbsphinx",
    "sphinx.ext.autodoc",
    "sphinx.ext.coverage",
    "sphinx.ext.githubpages",
    "sphinx.ext.ifconfig",
    "sphinx.ext.intersphinx",
    "sphinx.ext.linkcode",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx.ext.todo",
    "sphinx_gallery.gen_gallery",
    "sphinx_issues",
    "sphinx_runpython.blocdefs.sphinx_exref_extension",
    "sphinx_runpython.blocdefs.sphinx_faqref_extension",
    "sphinx_runpython.blocdefs.sphinx_mathdef_extension",
    "sphinx_runpython.epkg",
    "sphinx_runpython.gdot",
    "sphinx_runpython.runpython",
    "matplotlib.sphinxext.plot_directive",
]

if has_dvisvgm():
    extensions.append("sphinx.ext.imgmath")
    imgmath_image_format = "svg"
elif has_dvipng():
    extensions.append("sphinx.ext.pngmath")
    imgmath_image_format = "png"
else:
    extensions.append("sphinx.ext.mathjax")

templates_path = ["_templates"]
html_logo = "_static/project_ico.png"
source_suffix = ".rst"
master_doc = "index"
project = "teachcompute"
copyright = "2023-2024, Xavier Dupré"
author = "Xavier Dupré"
version = __version__
release = __version__
language = "fr"
exclude_patterns = ["auto_examples/*.ipynb"]
pygments_style = "sphinx"
todo_include_todos = True
nbsphinx_execute = "never"

html_theme = "furo"
html_theme_path = ["_static"]
html_theme_options = {}
html_sourcelink_suffix = ""
html_static_path = ["_static"]

issues_github_path = "sdpython/teachcompute"

nbsphinx_prolog = """

.. _nbl-{{ env.doc2path(env.docname, base=None).replace("/", "-").split(".")[0] }}:

"""

nbsphinx_epilog = """
----

`Notebook on github <https://github.com/sdpython/teachcompute/tree/main/_doc/{{ env.doc2path(env.docname, base=None) }}>`_
"""

# The following is used by sphinx.ext.linkcode to provide links to github
linkcode_resolve = make_linkcode_resolve(
    "teachcompute",
    (
        "https://github.com/sdpython/teachcompute/"
        "blob/{revision}/{package}/"
        "{path}#L{lineno}"
    ),
)

latex_elements = {
    "papersize": "a4",
    "pointsize": "10pt",
    "title": project,
}

mathjax3_config = {"chtml": {"displayAlign": "left"}}

intersphinx_mapping = {
    "IPython": ("https://ipython.readthedocs.io/en/stable/", None),
    "matplotlib": ("https://matplotlib.org/", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable/", None),
    "python": (f"https://docs.python.org/{sys.version_info.major}", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/reference", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
}

# Check intersphinx reference targets exist
nitpicky = True
# See also scikit-learn/scikit-learn#26761
nitpick_ignore = [
    ("py:class", "False"),
    ("py:class", "True"),
    ("py:class", "pipeline.Pipeline"),
    ("py:class", "default=sklearn.utils.metadata_routing.UNCHANGED"),
    ("py:class", "sklearn.ensemble.RandomForestRegressor"),
    ("py:class", "unittest.case.TestCase"),
]

nitpick_ignore_regex = [
    ("py:class", ".*numpy[.].*"),
    ("py:func", ".*[.]PyCapsule[.].*"),
    ("py:func", ".*numpy[.].*"),
    ("py:func", ".*scipy[.].*"),
]

sphinx_gallery_conf = {
    # path to your examples scripts
    "examples_dirs": os.path.join(os.path.dirname(__file__), "examples"),
    # path where to save gallery generated examples
    "gallery_dirs": "auto_examples",
}

# next

preamble = """
\\usepackage{etex}
\\usepackage{fixltx2e} % LaTeX patches, \\textsubscript
\\usepackage{cmap} % fix search and cut-and-paste in Acrobat
\\usepackage[raccourcis]{fast-diagram}
\\usepackage{titlesec}
\\usepackage{amsmath}
\\usepackage{amssymb}
\\usepackage{amsfonts}
\\usepackage{graphics}
\\usepackage{epic}
\\usepackage{eepic}
%\\usepackage{pict2e}
%%% Redefined titleformat
\\setlength{\\parindent}{0cm}
\\setlength{\\parskip}{1ex plus 0.5ex minus 0.2ex}
\\newcommand{\\hsp}{\\hspace{20pt}}
\\newcommand{\\acc}[1]{\\left\\{#1\\right\\}}
\\newcommand{\\cro}[1]{\\left[#1\\right]}
\\newcommand{\\pa}[1]{\\left(#1\\right)}
\\newcommand{\\R}{\\mathbb{R}}
\\newcommand{\\HRule}{\\rule{\\linewidth}{0.5mm}}
%\\titleformat{\\chapter}[hang]{\\Huge\\bfseries\\sffamily}{\\thechapter\\hsp}{0pt}{\\Huge\\bfseries\\sffamily}

\\usepackage[all]{xy}
\\newcommand{\\vecteur}[2]{\\pa{#1,\\dots,#2}}
\\newcommand{\\N}[0]{\\mathbb{N}}
\\newcommand{\\indicatrice}[1]{ {1\\!\\!1}_{\\acc{#1}} }
\\newcommand{\\infegal}[0]{\\leqslant}
\\newcommand{\\supegal}[0]{\\geqslant}
\\newcommand{\\ensemble}[2]{\\acc{#1,\\dots,#2}}
\\newcommand{\\fleche}[1]{\\overrightarrow{ #1 }}
\\newcommand{\\intervalle}[2]{\\left\\{#1,\\cdots,#2\\right\\}}
\\newcommand{\\independant}[0]{\\perp \\!\\!\\! \\perp}
\\newcommand{\\esp}{\\mathbb{E}}
\\newcommand{\\espf}[2]{\\mathbb{E}_{#1}\\pa{#2}}
\\newcommand{\\var}{\\mathbb{V}}
\\newcommand{\\pr}[1]{\\mathbb{P}\\pa{#1}}
\\newcommand{\\loi}[0]{{\\cal L}}
\\newcommand{\\vecteurno}[2]{#1,\\dots,#2}
\\newcommand{\\norm}[1]{\\left\\Vert#1\\right\\Vert}
\\newcommand{\\norme}[1]{\\left\\Vert#1\\right\\Vert}
\\newcommand{\\scal}[2]{\\left<#1,#2\\right>}
\\newcommand{\\dans}[0]{\\rightarrow}
\\newcommand{\\partialfrac}[2]{\\frac{\\partial #1}{\\partial #2}}
\\newcommand{\\partialdfrac}[2]{\\dfrac{\\partial #1}{\\partial #2}}
\\newcommand{\\trace}[1]{tr\\pa{#1}}
\\newcommand{\\sac}[0]{|}
\\newcommand{\\abs}[1]{\\left|#1\\right|}
\\newcommand{\\loinormale}[2]{{\\cal N} \\pa{#1,#2}}
\\newcommand{\\loibinomialea}[1]{{\\cal B} \\pa{#1}}
\\newcommand{\\loibinomiale}[2]{{\\cal B} \\pa{#1,#2}}
\\newcommand{\\loimultinomiale}[1]{{\\cal M} \\pa{#1}}
\\newcommand{\\variance}[1]{\\mathbb{V}\\pa{#1}}
\\newcommand{\\intf}[1]{\\left\\lfloor #1 \\right\\rfloor}
"""

epkg_dictionary = {
    "AITemplate": "https://github.com/facebookincubator/AITemplate",
    "Anaconda": "https://www.anaconda.com/",
    "API REST": "https://fr.wikipedia.org/wiki/Representational_state_transfer",
    "AVX": "https://en.wikipedia.org/wiki/Advanced_Vector_Extensions",
    "black": "https://github.com/psf/black",
    "BLAS": "https://www.netlib.org/blas/",
    "blas": "https://www.netlib.org/blas/",
    "blis": "https://github.com/flame/blis",
    "branching": "https://en.wikipedia.org/wiki/Branch_(computer_science)",
    "C++": "https://fr.wikipedia.org/wiki/C%2B%2B",
    "cacher": "https://en.wikipedia.org/wiki/Cache_(computing)",
    "cloudpickle": "https://github.com/cloudpipe/cloudpickle",
    "cmake": "https://cmake.org/",
    "compiler directives": "https://cython.readthedocs.io/en/stable/src/userguide/source_files_and_compilation.html#compiler-directives",
    "concurrent.futures": "https://docs.python.org/3/library/concurrent.futures.html",
    "CPU": "https://fr.wikipedia.org/wiki/Processeur",
    "CUDA": "https://developer.nvidia.com/cuda-gpus",
    "Custom C++ and CUDA Extensions": "https://pytorch.org/tutorials/advanced/cpp_extension.html",
    "cython": "https://cython.org/",
    "DLPack": "https://github.com/dmlc/dlpack",
    "dot": "https://fr.wikipedia.org/wiki/DOT_(langage)",
    "DOT": "https://fr.wikipedia.org/wiki/DOT_(langage)",
    "Eigen": "https://eigen.tuxfamily.org/",
    "eigen": "https://eigen.tuxfamily.org/",
    "générateur": "https://fr.wikipedia.org/wiki/G%C3%A9n%C3%A9rateur_(informatique)",
    "git": "https://en.wikipedia.org/wiki/Git",
    "github": "https://github.com/",
    "GPU": "https://fr.wikipedia.org/wiki/Processeur_graphique",
    "itérateur": "https://fr.wikipedia.org/wiki/It%C3%A9rateur",
    "joblib": "https://joblib.readthedocs.io/en/stable/",
    "JSON": "https://en.wikipedia.org/wiki/JSON",
    "kubernetes": "https://kubernetes.io/fr/",
    "lapack": "https://www.netlib.org/lapack/",
    "mllib": "https://spark.apache.org/mllib/",
    "nogil": "https://cython.readthedocs.io/en/stable/src/userguide/external_C_code.html?highlight=nogil#declaring-a-function-as-callable-without-the-gil",
    "notebook": "https://jupyter-notebook-beginner-guide.readthedocs.io/en/latest/what_is_jupyter.html#notebook-document",
    "numba": "https://numba.pydata.org/",
    "numpy": (
        "https://www.numpy.org/",
        ("https://docs.scipy.org/doc/numpy/reference/generated/numpy.{0}.html", 1),
        ("https://docs.scipy.org/doc/numpy/reference/generated/numpy.{0}.{1}.html", 2),
    ),
    "nvidia-smi": "https://developer.nvidia.com/nvidia-system-management-interface",
    "onnx-extended": "https://sdpython.github.io/doc/onnx-extended/dev/",
    "openmp": "https://www.openmp.org/",
    "OpenMP": "https://www.openmp.org/",
    "pandas": (
        "https://pandas.pydata.org/pandas-docs/stable/",
        ("https://pandas.pydata.org/pandas-docs/stable/generated/pandas.{0}.html", 1),
        (
            "https://pandas.pydata.org/pandas-docs/stable/generated/pandas.{0}.{1}.html",
            2,
        ),
    ),
    "prange": "https://cython.readthedocs.io/en/stable/src/userguide/parallelism.html",
    "processus": "https://en.wikipedia.org/wiki/Process_(computing)",
    "programmation impérative": "https://fr.wikipedia.org/wiki/Programmation_imp%C3%A9rative",
    "programmation fonctionnelle": "https://fr.wikipedia.org/wiki/Programmation_fonctionnelle",
    "protobuf": "https://protobuf.dev/",
    "psutil": "https://psutil.readthedocs.io/en/latest/",
    "pull request": "https://docs.github.com/fr/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/about-pull-requests",
    "pybind11": "https://github.com/pybind/pybind11",
    "pypi": "https://pypi.org/",
    "PyPi": "https://pypi.org/",
    "pyspark": "https://spark.apache.org/",
    "python": "https://www.python.org/",
    "Python": "https://www.python.org/",
    "Python C API": "https://docs.python.org/3/c-api/index.html",
    "pytorch": "https://pytorch.org/docs/stable/",
    "ruff": "https://github.com/astral-sh/ruff",
    "scikit-learn": "https://scikit-learn.org",
    "sérialisation": "https://fr.wikipedia.org/wiki/S%C3%A9rialisation",
    "setup.py": "https://docs.python.org/fr/3.11/distutils/setupscript.html",
    "spark": "https://spark.apache.org/",
    "Spark": "https://spark.apache.org/",
    "sphinx": "https://www.sphinx-doc.org/en/master/",
    "std": (
        "https://en.cppreference.com/w/cpp/header",
        ("https://en.cppreference.com/w/cpp/container/{0}", 1),
        ("https://en.cppreference.com/w/cpp/container/{1}", 2),
    ),
    "std::vector": "https://en.cppreference.com/w/cpp/container/vector",
    "teachcompute": "https://sdpython.github.io/doc/teachcompute/dev/",
    "teachpyx": "https://sdpython.github.io/doc/teachpyx/dev/",
    "threads": "https://fr.wikipedia.org/wiki/Thread_(informatique)",
    "transformers": "https://github.com/huggingface/transformers",
    "Triton": "https://openai.com/research/triton",
    "tqdm": "https://tqdm.github.io/",
    "TVM": "https://tvm.apache.org/",
    "ujson": "https://github.com/ultrajson/ultrajson",
    "Visual Studio Code": "https://code.visualstudio.com/",
    "viz.js": "https://github.com/mdaines/viz-js",
    "vscode-python": "https://code.visualstudio.com/docs/languages/python",
    "X-tree": "https://en.wikipedia.org/wiki/X-tree",
    "XML": "https://fr.wikipedia.org/wiki/Extensible_Markup_Language",
    "wikipedia dumps": "https://dumps.wikimedia.org/frwiki/latest/",
}

imgmath_latex_preamble = preamble
latex_elements["preamble"] = imgmath_latex_preamble
