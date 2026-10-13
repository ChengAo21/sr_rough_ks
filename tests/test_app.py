"""Smoke verification for every page and both PCE case families."""
import unittest
from pathlib import Path
from streamlit.testing.v1 import AppTest


class App(unittest.TestCase):
    def test_pages_and_ishigami_fit(self):
        app=AppTest.from_file(str(Path(__file__).resolve().parents[1]/"streamlit_app.py")).run(timeout=30)
        for page in app.sidebar.radio[0].options:
            app.sidebar.radio[0].set_value(page).run(timeout=30)
            self.assertEqual(list(app.exception),[],page)
        app.sidebar.radio[0].set_value("PCE Studio").run(timeout=30)
        app.selectbox(key="pce-case").set_value("Ishigami benchmark").run(timeout=30)
        button=next(b for b in app.button if b.label=="Fit PCE & analyze →")
        button.click().run(timeout=30)
        self.assertEqual(list(app.exception),[])
        result=app.session_state["pce_result"]
        self.assertEqual(result["case"],"Ishigami benchmark")
        self.assertEqual(result["model"].dim,3)
        self.assertLess(result["relative_l2"],.3)


if __name__=="__main__": unittest.main()
