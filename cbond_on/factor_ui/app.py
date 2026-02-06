from __future__ import annotations

from pathlib import Path

from flask import Flask, jsonify, request
from flask import render_template

from cbond_on.core.config import load_config_file
from cbond_on.factor_ui.service import (
    SimpleCache,
    analyze_single_factor,
    build_factor_index,
    load_factor_index,
    save_factor_index,
    summarize_factors,
)
from cbond_on.factors.spec import FactorSpec


def create_app() -> Flask:
    app = Flask(__name__)
    cache = SimpleCache()

    @app.get("/api/health")
    def health():
        return jsonify({"status": "ok"})

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.get("/api/index")
    def factor_index():
        cfg = load_config_file("factor_batch")
        paths = load_config_file("paths")
        panel_name = cfg.get("panel_name")
        window_minutes = int(cfg.get("window_minutes", 15))
        factor_root = Path(paths["factor_data_root"])
        index_path = factor_root / "factor_index.json"
        index = load_factor_index(index_path)
        if index is None:
            specs = [
                FactorSpec(
                    name=item["name"],
                    factor=item["factor"],
                    params=item.get("params", {}),
                    output_col=item.get("output_col"),
                )
                for item in cfg.get("factors", [])
            ]
            built = build_factor_index(
                factor_root=factor_root,
                panel_name=panel_name,
                window_minutes=window_minutes,
                specs=specs,
            )
            save_factor_index(built, index_path)
            index = load_factor_index(index_path)
        return jsonify(index or {})

    @app.post("/api/index/build")
    def build_index():
        cfg = load_config_file("factor_batch")
        paths = load_config_file("paths")
        panel_name = cfg.get("panel_name")
        window_minutes = int(cfg.get("window_minutes", 15))
        factor_root = Path(paths["factor_data_root"])
        index_path = factor_root / "factor_index.json"
        specs = [
            FactorSpec(
                name=item["name"],
                factor=item["factor"],
                params=item.get("params", {}),
                output_col=item.get("output_col"),
            )
            for item in cfg.get("factors", [])
        ]
        built = build_factor_index(
            factor_root=factor_root,
            panel_name=panel_name,
            window_minutes=window_minutes,
            specs=specs,
        )
        save_factor_index(built, index_path)
        return jsonify({"status": "ok", "index_path": str(index_path)})

    @app.post("/api/factor/analysis")
    def factor_analysis():
        payload = request.get_json(force=True) or {}
        try:
            result = analyze_single_factor(payload, cache=cache)
        except Exception as exc:
            return jsonify({"error": str(exc)}), 400
        return jsonify(result)

    @app.post("/api/factors/summary")
    def factors_summary():
        payload = request.get_json(force=True) or {}
        try:
            result = summarize_factors(payload, cache=cache)
        except Exception as exc:
            return jsonify({"error": str(exc)}), 400
        return jsonify(result)

    return app
