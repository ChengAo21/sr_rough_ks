import contextlib
import io
import unittest
from unittest.mock import patch
import numpy as np

from roughlab.models import load_data, published_predictions, drag_model, factors, metrics
from roughlab.pce import PolyChaos, fit_pce
from roughlab.symbolic import run_gp


class Numerics(unittest.TestCase):
    def test_data_mapping_and_paper_metrics(self):
        df=published_predictions()
        self.assertEqual(len(df),104)
        self.assertEqual(int((df.label=="DNSLC").sum()),8)
        self.assertAlmostEqual(df.iloc[0].Po,.5971)
        self.assertAlmostEqual(df.iloc[0].target,.0644/.0219)
        m=metrics(df.target,df.prediction)
        self.assertAlmostEqual(m["MARE (%)"],9.597889514948003,places=9)
        self.assertAlmostEqual(m["Max ARE (%)"],30.792371960041336,places=9)

    def test_continuous_model_limits(self):
        point=np.array([[.634,.517,0.,0.,.159]])
        self.assertTrue(np.isfinite(drag_model(point)).all())
        self.assertAlmostEqual(factors(point)[0,2],1.)
        self.assertAlmostEqual(factors(point)[0,3],.308)

    def test_exact_uniform_pce_moments_and_interactions(self):
        with contextlib.redirect_stdout(io.StringIO()):
            model=PolyChaos(2,["U","U"],[[-1,1],[-1,1]])
            points=model.create_latin_hypercube_samples(100,42)
            fun=lambda x:2+3*x[:,0]+2*x[:,0]*x[:,1]
            model.regression_fit(points,fun(points));model.norm_fit()
        valid=np.random.default_rng(3).uniform(-1,1,(80,2))
        np.testing.assert_allclose(model.evaluate(valid),fun(valid),rtol=1e-11,atol=1e-11)
        self.assertAlmostEqual(model.mu,2.)
        self.assertAlmostEqual(model.sigma**2,31/9)
        self.assertAlmostEqual(model.sobol([1]),27/31)
        self.assertAlmostEqual(model.sobol([1,2]),4/31)

    def test_exact_normal_pce(self):
        with contextlib.redirect_stdout(io.StringIO()):
            model=PolyChaos(2,["N","N"],[[1,2],[.5,3]])
            points=model.create_latin_hypercube_samples(100,42)
            z=(points-np.array([1,.5]))/np.array([2,3])
            y=2+3*z[:,0]+.5*(z[:,1]**2-1)
            model.regression_fit(points,y);model.norm_fit()
        np.testing.assert_allclose(model.evaluate(points),y,rtol=1e-11,atol=1e-11)
        self.assertAlmostEqual(model.mu,2.)
        self.assertAlmostEqual(model.sigma**2,9.5)

    def test_pce_rejects_invalid_distributions(self):
        with self.assertRaises(ValueError): fit_pce("Ishigami benchmark",2,100,42,["U"]*3,[[1,-1],[-1,1],[-1,1]])
        with self.assertRaises(ValueError): fit_pce("Ishigami benchmark",6,10,42,["U"]*3,[[-1,1]]*3)

    def test_gp_does_not_use_test_targets(self):
        config=dict(seed=42,template=True,unary=["tanh","square"],binary=["+","*"],population=24,iterations=5,maxsize=30,timeout=60,parsimony=.003)
        first=run_gp(config)
        changed=load_data();changed.loc[changed.label=="DNSLC","target"]*=1000
        with patch("roughlab.symbolic.load_data",return_value=changed): second=run_gp(config)
        self.assertEqual(first["equation"],second["equation"])
        self.assertEqual(first["history"],second["history"])
        self.assertNotEqual(first["test_metrics"],second["test_metrics"])


if __name__=="__main__": unittest.main()
