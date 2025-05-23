import yaml
import os
from dotenv import load_dotenv

load_dotenv()


def load_config(path: str = None) -> dict:
    if path is None:
        path = os.path.join(os.path.dirname(__file__), "..", "config.yaml")
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    def resolve_env(val):
        if isinstance(val, str) and val.startswith("${") and val.endswith("}"):
            return os.getenv(val[2:-1], "")
        return val

    def resolve_nested(d):
        if isinstance(d, dict):
            return {k: resolve_nested(v) for k, v in d.items()}
        elif isinstance(d, list):
            return [resolve_nested(v) for v in d]
        return resolve_env(d)

    return resolve_nested(config)


# test hàm
if __name__ == "__main__":
    config = load_config()
    print(config)
    # print(config["model"]["name"])
    # print(config["model"]["params"]["temperature"])
    # print(config["model"]["params"]["max_tokens"])
    # print(config["retriever"]["index_path"])
    # print(config["retriever"]["metadata_path"])
