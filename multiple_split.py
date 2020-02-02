def multiple_split(string, substring):
    for sub in substring:
        if isinstance(string, str):
            string = string.split(sub)
        elif isinstance(string, list):
            string = sum([st.split(sub) for st in string], [])
    string = [i for i in string if i != '']
    return string