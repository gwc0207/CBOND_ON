from __future__ import annotations

from datetime import date
import math

import numpy as np
import pandas as pd
import torch

from cbond_on.infra.model.adapters import KlineImageAdapter, KlineTimeFrequencyAdapter, build_adapter
from cbond_on.infra.model.impl.torch_image import KlineImageCNN, KlineTimeFrequencyFusionCNN
from cbond_on.infra.model.runners.train_kline_image import (
    _extract_day_samples_from_frame,
    render_candlestick_batch,
)
from cbond_on.infra.model.runners.train_kline_time_frequency import (
    render_cwt_batch,
    render_stft_batch,
)


def test_render_candlestick_batch_uses_fixed_scale() -> None:
    flat = np.zeros((2, 120, 4), dtype=np.float32)
    flat[0, :, :] = 0.01
    flat[1, :, :] = 0.05

    images = render_candlestick_batch(
        torch.from_numpy(flat),
        image_height=64,
        price_limit=0.10,
    )

    first_y = torch.nonzero(images[0].sum(dim=0), as_tuple=False)[:, 0].float().mean()
    second_y = torch.nonzero(images[1].sum(dim=0), as_tuple=False)[:, 0].float().mean()
    assert second_y < first_y


def test_render_candlestick_batch_separates_up_and_down_bodies() -> None:
    ohlc = np.zeros((1, 2, 4), dtype=np.float32)
    ohlc[0, 0] = [0.00, 0.02, -0.01, 0.01]
    ohlc[0, 1] = [0.01, 0.02, -0.01, -0.005]

    image = render_candlestick_batch(
        torch.from_numpy(ohlc),
        image_height=32,
        price_limit=0.10,
    )[0]

    assert float(image[1, :, 0].sum()) > 0
    assert float(image[2, :, 0].sum()) == 0
    assert float(image[2, :, 1].sum()) > 0
    assert float(image[1, :, 1].sum()) == 0


def test_extract_day_samples_excludes_1130_bar_from_image() -> None:
    rows: list[dict] = []
    code = "110001.SH"
    previous = 100.0
    for minute in range(570, 691):
        close = 100.0 + (minute - 570) * 0.01
        if minute == 690:
            close = 150.0
        rows.append(
            {
                "code": code,
                "minute": minute,
                "open_price": close - 0.01,
                "high_price": close + 0.02,
                "low_price": close - 0.02,
                "close_price": close,
                "prev_close_price": previous,
                "twap": close,
            }
        )
        previous = close
    for minute in range(780, 785):
        rows.append(
            {
                "code": code,
                "minute": minute,
                "open_price": 102.0,
                "high_price": 102.0,
                "low_price": 102.0,
                "close_price": 102.0,
                "prev_close_price": 102.0,
                "twap": 102.0,
            }
        )
    for minute in range(890, 897):
        rows.append(
            {
                "code": code,
                "minute": minute,
                "open_price": 103.0,
                "high_price": 103.0,
                "low_price": 103.0,
                "close_price": 103.0,
                "prev_close_price": 103.0,
                "twap": 103.0,
            }
        )

    data = _extract_day_samples_from_frame(
        pd.DataFrame(rows),
        trade_day=date(2026, 7, 15),
        allowed_codes={code},
        data_cfg={
            "morning_start": "09:30",
            "morning_end_exclusive": "11:30",
            "buy_start": "13:00",
            "buy_end_exclusive": "13:05",
            "sell_start": "14:50",
            "sell_end_exclusive": "14:57",
            "session_end_exclusive": "15:00",
            "min_morning_bars": 120,
        },
        buy_cost_bps=1.0,
        sell_cost_bps=1.2,
    )

    assert data.ohlc.shape == (1, 120, 4)
    expected_last = np.log((100.0 + 119 * 0.01) / 100.0)
    assert np.isclose(data.ohlc[0, -1, 3], expected_last, atol=1e-4)
    assert np.isclose(data.raw_return[0], 103.0 / 102.0 - 1.0 - 0.00022)


def test_kline_image_model_and_adapter_contract() -> None:
    model = KlineImageCNN(base_channels=8)
    output = model(torch.zeros((3, 3, 64, 120), dtype=torch.float32))
    assert output.shape == (3,)
    assert isinstance(build_adapter("kline_image_cnn"), KlineImageAdapter)


def test_time_frequency_renderers_preserve_batch_and_direction() -> None:
    ohlc = torch.zeros((2, 120, 4), dtype=torch.float32)
    path = torch.linspace(0.0, 0.02, 120)
    ohlc[0, :, 3] = path
    ohlc[1, :, 3] = -path

    stft = render_stft_batch(ohlc, n_fft=32, win_length=32, hop_length=4)
    cwt = render_cwt_batch(ohlc, num_scales=12, min_period=2.0, max_period=32.0, pad=32)

    assert stft.shape == (2, 3, 17, 31)
    assert cwt.shape == (2, 3, 12, 120)
    assert torch.isfinite(stft).all()
    assert torch.isfinite(cwt).all()
    assert not torch.allclose(cwt[0, 0], cwt[1, 0])
    assert torch.allclose(cwt[0, 2], cwt[1, 2], atol=1e-5)


def test_time_frequency_fusion_model_and_adapter_contract() -> None:
    model = KlineTimeFrequencyFusionCNN(base_channels=8)
    output = model(
        torch.zeros((3, 3, 64, 120), dtype=torch.float32),
        torch.zeros((3, 3, 16, 120), dtype=torch.float32),
    )
    assert output.shape == (3,)
    assert isinstance(build_adapter("kline_time_frequency_cnn"), KlineTimeFrequencyAdapter)


def test_cwt_localizes_known_periods() -> None:
    configured_periods = np.logspace(np.log10(2.0), np.log10(48.0), 16)
    ohlc = torch.zeros((2, 120, 4), dtype=torch.float32)
    timeline = torch.arange(120, dtype=torch.float32)
    returns = torch.stack(
        [
            0.002 * torch.sin(2.0 * math.pi * timeline / 8.0),
            0.002 * torch.sin(2.0 * math.pi * timeline / 32.0),
        ]
    )
    ohlc[:, :, 3] = torch.cumsum(returns, dim=1)

    cwt = render_cwt_batch(
        ohlc,
        num_scales=16,
        min_period=2.0,
        max_period=48.0,
        pad=60,
    )
    peak_periods = configured_periods[cwt[:, 2].mean(dim=2).argmax(dim=1).numpy()]

    assert abs(np.log(peak_periods[0] / 8.0)) < 0.15
    assert abs(np.log(peak_periods[1] / 32.0)) < 0.15
