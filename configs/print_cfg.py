import yaml


def print_yaml_keys_values(data, parent_key=""):
    if isinstance(data, dict):
        for key, value in data.items():
            full_key = f"{parent_key}.{key}" if parent_key else key
            print(f"{full_key}: {value}")
            # print_yaml_keys_values(value, full_key)
    elif isinstance(data, list):
        for index, item in enumerate(data):
            full_key = f"{parent_key}[{index}]"
            print(f"{full_key}: {item}")
            # print_yaml_keys_values(item, full_key)


yaml_file = "configs/citeseer/1-shot/gptt.yaml"
with open(yaml_file, "r", encoding="utf-8") as file:
    yaml_data = yaml.safe_load(file)

print_yaml_keys_values(yaml_data)
