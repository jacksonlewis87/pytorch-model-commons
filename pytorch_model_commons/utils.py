from dataclasses import fields


def dict_to_dataclass(dataclass_type, dict_obj):
    valid_fields = {field.name for field in fields(dataclass_type)}
    filtered_dict = {key: value for key, value in dict_obj.items() if key in valid_fields}

    return dataclass_type(**filtered_dict)
