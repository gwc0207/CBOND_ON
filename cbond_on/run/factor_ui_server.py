from __future__ import annotations

from cbond_on.factor_ui.app import create_app


def main() -> None:
    app = create_app()
    app.run(host="0.0.0.0", port=5001, debug=False)


if __name__ == "__main__":
    main()
