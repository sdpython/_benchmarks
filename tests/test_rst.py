"""
@brief      test log(time=93s)
"""
import os
import unittest
from pyquickhelper.pycode import (
    get_temp_folder,
    skipif_travis, skipif_appveyor
)
from pyquickhelper.loghelper import noLOG
from pyquickhelper.pycode import ExtTestCase
from pyquickhelper.helpgen import rst2html


class TestRst(ExtTestCase):

    def common_test(self, name):
        preamble = """
            \\newcommand{\\acc}[1]{\\left\\{#1\\right\\}}
            \\newcommand{\\abs}[1]{\\left\\{#1\\right\\}}
            \\newcommand{\\cro}[1]{\\left[#1\\right]}
            \\newcommand{\\pa}[1]{\\left(#1\\right)}
            \\newcommand{\\girafedec}[3]{ \\begin{array}{ccccc} #1 &=& #2 &+& #3 \\\\ a' &=& a &-& o  \\end{array}}
            \\newcommand{\\vecteur}[2]{\\pa{#1,\\dots,#2}}
            \\newcommand{\\R}[0]{\\mathbb{R}}
            \\newcommand{\\N}[0]{\\mathbb{N}}
            \\newcommand{\\indicatrice}[1]{\\mathbf{1\\!\\!1}_{\\acc{#1}}}
            \\usepackage[all]{xy}
            \\newcommand{\\infegal}[0]{\\leqslant}
            \\newcommand{\\supegal}[0]{\\geqslant}
            \\newcommand{\\ensemble}[2]{\\acc{#1,\\dots,#2}}
            \\newcommand{\\fleche}[1]{\\overrightarrow{ #1 }}
            \\newcommand{\\esp}{\\mathbb{E}}
            """

        temp = get_temp_folder(__file__, "temp_rst_" + name.split('.')[0])
        clone = 0
        links = dict(onnx="http", python='python')
        links["scikit-learn"] = "skl"

        doc = os.path.join(temp, "..", "..", "_doc", "onnx", name)

        with open(doc, "r", encoding="utf-8") as f:
            content = f.read()
        content = content.replace(":ref:", "     ")
        text = rst2html(content, outdir=temp, writer="rst",
                        imgmath_latex_preamble=preamble, layout="docutils",
                        extlinks=dict(issue=('https://link/%s',
                                             'issue %s on GitHub')),
                        epkg_dictionary=links,
                        document_name=doc, destination_path=temp,
                        fLOG=noLOG)
        return text

    @skipif_travis("needs latex")
    @skipif_appveyor("needs latex")
    def test_rst_onnxruntime_datasets_num(self):
        text = self.common_test("onnxruntime_datasets_num.rst")
        self.assertIn('+---------', text)


if __name__ == "__main__":
    unittest.main()
