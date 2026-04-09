from __future__ import annotations

import pandas as pd
import torch

from cbond_on.core.config import load_config_file, resolve_output_path

from .base import BaseModel
from .score_io import load_scores_by_date
from .impl.lob.lob_st import LOBSpatioTemporalModel


class FileScoreModel(BaseModel):
    def __init__(self, score_path: str) -> None:
        self._cache = load_scores_by_date(score_path)

    def predict(self, data: pd.DataFrame) -> pd.Series:
        if "trade_date" not in data.columns:
            raise KeyError("score model requires trade_date column")
        day = pd.to_datetime(data["trade_date"].iloc[0]).date()
        scores = self._cache.get(day, pd.DataFrame())
        if scores.empty:
            return pd.Series(index=data.index, dtype=float)
        merged = data.merge(scores, on="code", how="left")
        return merged["score"].astype(float)


def build_model(cfg: dict) -> BaseModel:
    paths_cfg = load_config_file("paths")
    results_root = paths_cfg["results_root"]
    model_type = str(cfg.get("model_type", "custom")).lower()
    if model_type == "file":
        score_path = cfg.get("score_output")
        if not score_path:
            raise ValueError("model_type=file requires score_output path")
        resolved_score_path = resolve_output_path(
            score_path,
            default_path=f"{results_root}/scores/file_model",
            results_root=results_root,
        )
        return FileScoreModel(str(resolved_score_path))
    if model_type == "lob_st":
        params = cfg.get("params") or {}
        weights_path = cfg.get("weights_path")
        if not weights_path:
            raise ValueError("model_type=lob_st requires weights_path")
        resolved_weights_path = resolve_output_path(
            weights_path,
            default_path=f"{results_root}/models/lob_st_default/model.pt",
            results_root=results_root,
        )
        model = LOBSpatioTemporalModel(
            depth_levels=int(params.get("depth_levels", 10)),
            rbf_num_bases=int(params.get("rbf_num_bases", 16)),
            rbf_sigma=float(params.get("rbf_sigma", 0.5)),
            lstm_hidden_size=int(params.get("lstm_hidden_size", 256)),
            lstm_num_layers=int(params.get("lstm_num_layers", 1)),
        )
        state = torch.load(str(resolved_weights_path), map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        model.load_state_dict(state)
        model.eval()

        class _LobStWrapper(BaseModel):
            def __init__(self, inner: LOBSpatioTemporalModel):
                self.inner = inner

            def predict(self, data: pd.DataFrame) -> pd.Series:
                raise NotImplementedError(
                    "LOB ST model requires tensor input; provide a custom predictor."
                )

        return _LobStWrapper(model)
    raise ValueError(f"unsupported model_type: {model_type}")
