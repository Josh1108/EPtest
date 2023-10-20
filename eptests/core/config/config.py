from jiant.utils.python import io


def read_cfg(path):
    return io.read_json_or_yml(path)


def write_cfg(data, path):
    return io.write_json_yml(data, path)


# print(read_cfg('/Users/Josh/Desktop/Projects/eptests/eptests/core/config/dummy_jiant.yaml'))
