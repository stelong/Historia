from pathlib import Path


def posix_path(*args: str) -> str:
    return Path().joinpath(*args).as_posix()


TOP_DIR = Path(__file__).parent.parent.parent.resolve()
SOURCE_DIR = posix_path(TOP_DIR.as_posix(), "Historia")
RESOURCES_DIR = posix_path(SOURCE_DIR, "resources")

RESOURCES_HEADERS_DIR = posix_path(RESOURCES_DIR, "headers")
RESOURCES_PARAMETERS_DIR = posix_path(RESOURCES_DIR, "parameters")
