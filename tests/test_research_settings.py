import contextlib
import io
import unittest
import numpy as np
from roughlab.models import BOUNDS, pairwise_response
from roughlab.charts import surface_grid, pce_surface_data
from roughlab.pce import PolyChaos, mean_factor, pairwise_projection_coefficients
from roughlab.sr_config import PAPER_OPTIONS, NESTING, preset_config, pysr_options
from roughlab.symbolic import complexity, respects_nesting, run_gp


class ResearchSettings(unittest.TestCase):
    def test_pairwise_response_is_the_two_factor_product(self):
        points=np.array([[.2,.2,-1.,.01,.08],[.6,.517,0.,.1,.3],[.9,.8,1.4,.4,.7]])
        ex,sk=points[:,1],points[:,2]
        ratio=np.divide(sk,np.tanh(sk),out=np.ones_like(sk),where=sk!=0)
        expected=np.tanh((ex-.084)/.294)*(np.tanh(sk)+ratio)
        np.testing.assert_allclose(pairwise_response(points,1,2),expected)
        changed=points.copy();changed[:,[0,3,4]]=[[.8,.4,.7],[.2,.01,.08],[.6,.1,.3]]
        np.testing.assert_array_equal(pairwise_response(points,1,2),pairwise_response(changed,1,2))
        xx,yy,_,z,_,_=surface_grid(resolution=8)
        np.testing.assert_allclose(z,np.tanh((xx-.084)/.294)*(np.tanh(yy)+yy/np.tanh(yy)))

    def test_paper_preset_passes_the_original_pysr_settings(self):
        config=preset_config("Paper settings")
        actual=pysr_options(config,"run")
        for key in ["niterations","populations","population_size","ncycles_per_iteration","maxsize",
                    "complexity_of_constants","unary_operators","binary_operators","nested_constraints",
                    "precision","model_selection","elementwise_loss","parallelism","progress","verbosity"]:
            self.assertEqual(actual[key],PAPER_OPTIONS[key],key)
        self.assertEqual(actual["maxsize"],60)
        self.assertEqual(actual["complexity_of_constants"],3)
        self.assertIsNone(actual["timeout_in_seconds"])
        self.assertFalse(actual["deterministic"])
        config.update(maxsize=28,constant_complexity=5,parallelism="serial")
        config["nested_constraints"]["exp"]["square"]=0
        custom=pysr_options(config,"run")
        self.assertEqual(custom["maxsize"],28)
        self.assertEqual(custom["complexity_of_constants"],5)
        self.assertEqual(custom["nested_constraints"]["exp"]["square"],0)
        self.assertTrue(custom["deterministic"])

    def test_gp_nesting_checks_all_branches(self):
        x=("v",0)
        for tree in [("exp",("square",x)),("cos",("square",x)),("square",("exp",x)),
                     ("exp",("+",("square",x),("square",x)))]:
            self.assertTrue(respects_nesting(tree,NESTING),tree)
        for tree in [("exp",("exp",x)),("tanh",("tanh",x)),("square",("square",x)),
                     ("exp",("+",x,("cos",x))),("exp",("square",("exp",x)))]:
            self.assertFalse(respects_nesting(tree,NESTING),tree)
        self.assertEqual(complexity(("+",x,("c",2.)),3),5)
        self.assertEqual(complexity(("+",x,("c",2.)),5),7)

    def test_live_gp_obeys_nesting_and_weighted_size(self):
        config=preset_config();config.update(engine="Python GP",iterations=8,population=32,maxsize=25,constant_complexity=4)
        result=run_gp(config)
        trees=result["structure"]["factors"]
        self.assertTrue(all(respects_nesting(tree,config["nested_constraints"]) for tree in trees))
        cost=sum(complexity(tree,4) for tree in trees)+5
        self.assertLessEqual(cost,25)
        self.assertTrue(all(row["complexity"]<=25 for row in result["frontier"]))
        self.assertEqual(result["config"]["constant_complexity"],4)

    def test_factor_mean_matches_analytical_exponential_moment(self):
        with contextlib.redirect_stdout(io.StringIO()):
            model=PolyChaos(2,["N"]*5,[[0,1]]*4+[[.159,.107]])
        self.assertAlmostEqual(mean_factor(model,4),np.exp(2*.159+2*.107**2),places=12)

    def test_pairwise_projection_removes_the_other_dimensions(self):
        with contextlib.redirect_stdout(io.StringIO()):
            model=PolyChaos(2,["U"]*5,BOUNDS)
        scale=np.prod([mean_factor(model,i) for i in (0,3,4)])
        model.coeff=np.zeros(model.nt)
        values={(0,0,0,0,0):10,(0,1,0,0,0):2,(0,0,1,0,0):-3,
                (0,1,1,0,0):4,(1,0,0,0,0):900,(1,1,0,0,0):100}
        for i,alpha in enumerate(model.multi_index): model.coeff[i]=scale*values.get(tuple(alpha),0)
        full=model.coeff.copy()
        projected=pairwise_projection_coefficients(model)
        model.coeff=projected
        points=np.random.default_rng(19).uniform(np.array(BOUNDS)[:,0],np.array(BOUNDS)[:,1],(50,5))
        ex=2*(points[:,1]-BOUNDS[1][0])/(BOUNDS[1][1]-BOUNDS[1][0])-1
        sk=2*(points[:,2]-BOUNDS[2][0])/(BOUNDS[2][1]-BOUNDS[2][0])-1
        np.testing.assert_allclose(model.evaluate(points),10+2*ex-3*sk+4*ex*sk,atol=1e-12)
        model.coeff=full
        pce_surface_data({"model":model,"case":"Roughness drag","order":2},resolution=8)
        np.testing.assert_array_equal(model.coeff,full)


if __name__=="__main__": unittest.main()
