import os


def get_model_names() -> list[str]:
    """
    Returns a list of model names found in the "./models" directory relative to the location of this file.

    Returns:
        list[str]: A list of strings representing the names of subdirectories in the "./models" directory.
    """  # noqa: E501
    model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "./models"))
    return [f.name for f in os.scandir(model_dir) if f.is_dir()]
