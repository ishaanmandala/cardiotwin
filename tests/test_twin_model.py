import math

import pytest
import torch

from cardiotwin.constants import TARGETS, lv_mass_ase, target_spec
from cardiotwin.models.resnet1d import CardioTwinNet, ModelConfig, count_parameters
from cardiotwin.training.losses import multitask_loss
from cardiotwin.twin import build_profile, probability_abnormal, render_profile_text, structural_risk


@pytest.mark.parametrize("ef,label,sev", [
    (60, "Normal", 0), (51.9, "Mildly reduced", 1), (41, "Mildly reduced", 1), (40, "Moderately reduced", 2),
    (30, "Moderately reduced", 2), (29.9, "Severely reduced", 3),
])
def test_ef_categories_follow_ase(ef, label, sev):
    assert TARGETS["lvef"].categorize(ef, "M") == (label, sev)


def test_sex_specific_ranges_and_unknown_sex_is_lenient():
    spec = TARGETS["lvedd"]
    assert spec.categorize(55, "F")[1] == 1  # mildly enlarged for a woman
    assert spec.categorize(55, "M")[1] == 0  # normal for a man
    assert spec.categorize(55, None)[1] == 0  # unknown sex: never over-call
    assert TARGETS["lvef"].categorize(53, "F")[1] == 1 and TARGETS["lvef"].categorize(53, "M")[1] == 0


def test_lv_mass_formula():
    # ASE example: IVSd 1.0, LVIDd 4.8, PWd 1.0 cm -> ~ 177 g
    assert lv_mass_ase(48, 10, 10) == pytest.approx(0.8 * 1.04 * (6.8**3 - 4.8**3) + 0.6)


def test_probability_abnormal_behaves():
    p_low = probability_abnormal("lvef", 65, 5, "M")
    p_high = probability_abnormal("lvef", 35, 5, "M")
    assert p_low < 0.1 < 0.9 < p_high
    assert probability_abnormal("lvedd", 70, 3, "M") > 0.99
    assert target_spec("something_new").label == "something_new"


def test_profile_and_text():
    preds = {"lvef": {"mean": 32, "sd": 6}, "lvedd": {"mean": 64, "sd": 4}, "ivsd": {"mean": 9, "sd": 1},
             "future_lvef": {"mean": 28, "sd": 7}, "ef_decline": {"mean": 0.62, "sd": 0}}
    prof = build_profile(preds, sex="M", truth={"lvef": 30}, future={"risk": {"label": "HF event", "probability": 0.3,
                                                                              "level": "HIGH"}})
    names = [i["target"] for i in prof["current"]]
    assert names == ["lvef", "lvedd", "ivsd"]
    assert prof["structural_risk"]["level"] == "HIGH"
    assert prof["current"][0]["truth"] == 30
    assert prof["current"][0]["low90"] == pytest.approx(32 - 1.6449 * 6)
    fut = {i["target"] for i in prof["future"]["measurements"]}
    assert fut == {"future_lvef", "ef_decline"}
    text = render_profile_text(prof)
    assert "CARDIOTWIN" in text and "FUTURE HEART" in text and "not a medical device" in text
    assert structural_risk([])["level"] == "UNKNOWN"


def _cfg(tab_dim=0, kinds=("regression", "regression", "binary")):
    return ModelConfig(n_leads=3, targets=["lvef", "lvedd", "ef_decline"], target_kinds=list(kinds), tab_dim=tab_dim,
                       width=8)


def test_model_shapes_and_tabular_requirement():
    net = CardioTwinNet(_cfg(tab_dim=2))
    out = net(torch.randn(4, 3, 2500), torch.randn(4, 2))
    assert out["mean"].shape == (4, 3) and out["log_var"].shape == (4, 3) and out["embedding"].shape == (4, 128)
    with pytest.raises(ValueError):
        net(torch.randn(2, 3, 2500))
    assert count_parameters(net) < 500_000
    assert CardioTwinNet(_cfg())(torch.randn(1, 3, 1000))["mean"].shape == (1, 3)  # any length works


def test_loss_ignores_missing_targets():
    torch.manual_seed(0)
    net = CardioTwinNet(_cfg())
    x = torch.randn(4, 3, 2500)
    out = net(x)
    y = torch.zeros(4, 3)
    mask = torch.tensor([[1, 0, 1]] * 4, dtype=torch.float32)
    loss, stats = multitask_loss(out, y, mask, torch.tensor([False, False, True]))
    loss.backward()
    grad_missing = net.mean_head.weight.grad[1].abs().sum().item()
    grad_present = net.mean_head.weight.grad[0].abs().sum().item()
    assert grad_missing == 0.0 and grad_present > 0
    assert math.isfinite(stats["loss"])
