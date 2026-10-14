"""Read the research configuration without executing its training script."""
import ast
from copy import deepcopy
from .models import ROOT


def _read_call(name):
    tree=ast.parse((ROOT/"results/SR_train.py").read_text(encoding="utf-8-sig"))
    call=next(node for node in ast.walk(tree) if isinstance(node,ast.Call)
              and isinstance(node.func,ast.Name) and node.func.id==name)
    values={}
    for keyword in call.keywords:
        try: values[keyword.arg]=ast.literal_eval(keyword.value)
        except (ValueError,TypeError): pass
    return values


PAPER_OPTIONS=_read_call("PySRRegressor")
PAPER_TEMPLATE=_read_call("TemplateExpressionSpec")
UNARY=PAPER_OPTIONS["unary_operators"]
BINARY=PAPER_OPTIONS["binary_operators"]
NESTING=PAPER_OPTIONS["nested_constraints"]


def preset_config(profile="Quick run"):
    options=PAPER_OPTIONS
    config=dict(profile=profile,engine="PySR",template=True,seed=42,
        iterations=options["niterations"],populations=options["populations"],
        population=options["population_size"],cycles=options["ncycles_per_iteration"],
        maxsize=options["maxsize"],constant_complexity=options["complexity_of_constants"],
        unary=list(UNARY),binary=list(BINARY),nested_constraints=deepcopy(NESTING),
        precision=options["precision"],model_selection=options["model_selection"],
        elementwise_loss=options["elementwise_loss"],parallelism=options["parallelism"],
        timeout=0,parsimony=.003)
    if profile=="Quick run":
        config.update(iterations=25,populations=4,cycles=100,timeout=60,parallelism="serial")
    elif profile!="Paper settings":
        raise ValueError(f"Unknown configuration preset: {profile}")
    return config


def normalize_config(config):
    result=preset_config(config.get("profile","Quick run"))
    result.update(deepcopy(config))
    if not set(result["unary"]).issubset(UNARY) or not set(result["binary"]).issubset(BINARY):
        raise ValueError("Choose operators from the original research configuration.")
    if not result["unary"] and not result["binary"]:
        raise ValueError("Select at least one operator.")
    if result["maxsize"]<10 or result["constant_complexity"]<1:
        raise ValueError("Maximum complexity must be at least 10 and constant complexity at least 1.")
    if result["engine"]=="Python GP":
        minimum=6+result["constant_complexity"] if result["template"] else 3+2*result["constant_complexity"]
        if result["maxsize"]<minimum:
            raise ValueError(f"Python GP needs maxsize at least {minimum} with this template and constant weight.")
    if result["parallelism"] not in ("serial","multithreading"):
        raise ValueError("Execution mode must be serial or multithreading.")
    if result["timeout"]<0:
        raise ValueError("The time limit cannot be negative.")
    result["nested_constraints"]={outer:{inner:int(limit) for inner,limit in limits.items()
        if inner in result["unary"] and int(limit)>=0}
        for outer,limits in result["nested_constraints"].items() if outer in result["unary"]}
    return result


def pysr_options(config,output_directory):
    config=normalize_config(config)
    return dict(niterations=config["iterations"],populations=config["populations"],
        population_size=config["population"],ncycles_per_iteration=config["cycles"],
        maxsize=config["maxsize"],complexity_of_constants=config["constant_complexity"],
        binary_operators=config["binary"],unary_operators=config["unary"],
        nested_constraints=config["nested_constraints"],precision=config["precision"],
        model_selection=config["model_selection"],elementwise_loss=config["elementwise_loss"],
        parallelism=config["parallelism"],deterministic=config["parallelism"]=="serial",
        random_state=config["seed"],timeout_in_seconds=config["timeout"] or None,
        progress=PAPER_OPTIONS["progress"],verbosity=PAPER_OPTIONS["verbosity"],
        output_directory=str(output_directory))
