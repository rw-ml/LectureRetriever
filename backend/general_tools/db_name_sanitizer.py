
def clean_name(name):
    name.replace("/", "_").replace("-", "_")
    return name