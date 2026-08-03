def generation_row(row):
    sanitized = dict(row)
    sanitized["bbox"] = [-1, -1, -1, -1]
    return sanitized