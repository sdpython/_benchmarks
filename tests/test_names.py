"""
@brief      test log(time=93s)
"""
import glob
import os
import unittest
import re
import pprint
from pyquickhelper.pycode import (
    get_temp_folder,
    skipif_travis, skipif_appveyor
)
from pyquickhelper.loghelper import noLOG
from pyquickhelper.pycode import ExtTestCase
from pyquickhelper.helpgen import rst2html
from mlprodict.tools.filename_helper import (
    extract_information_from_filename,
    make_readable_title
)


class TestNames(ExtTestCase):

    def test_check_names(self):
        this = os.path.abspath(os.path.dirname(__file__))
        doc = os.path.join(this, "..", "_doc", "onnx")
        fold_reg = os.path.join(this, "..", "onnx", "profiles_reg")
        fold_cls = os.path.join(this, "..")
        
        exp = {'AdaBoostClassifier', 'BernoulliNB', 'DecisionTreeClassifier',
               'GradientBoostingClassifier', 'LogisticRegression',
               'MLPClassifier', 'RandomForestClassifier', 'AdaBoostRegressor',
               'DecisionTreeRegressor', 'HistGradientBoostingRegressor',
               'KNeighborsRegressor', 'LinearRegression',
               'LogisticRegression', 'MLPRegressor', 'RandomForestRegressor',
               'RandomForestClassifier', 'SVR', 'SVC'}
        
        # Extract patterns.
        doc_files = glob.glob(os.path.join(doc, 'onnx_profiling_*.rst'))
        for name in doc_files:
            with open(name, "r", encoding="utf-8") as f:
                content = f.read()
            reg = re.compile("pattern = \\\"(onnx/profiles.*.svg)\\\"")
            pats = reg.findall(content)[0]
            
            pats = [os.path.join(fold_reg, pats),
                    os.path.join(fold_cls, pats)]
            svgs = glob.glob(pats[0]) + glob.glob(pats[1])
            
            if len(svgs) == 0 and 'knn' not in name:
                raise AssertionError(
                    "Unable to find any file for '{}' in\n{}.".format(
                        name, "\n".join(pats)))

            for svg in svgs:
                info = extract_information_from_filename(svg)
                title = make_readable_title(info)
                model = title.split()[0]
                if model not in exp:
                    pinfo = pprint.pformat(info)
                    raise AssertionError("Wrong title '{}' for '{}', info:\n{}.".format(
                        title, os.path.split(svg)[-1], pinfo))


if __name__ == "__main__":
    unittest.main()
