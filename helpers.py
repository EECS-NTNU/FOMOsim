def normalize_list(parameter_list: [float], zero_divide_value=0.0):
    min_value = min(parameter_list)
    max_value = max(parameter_list)

    return (
        [
            (parameter - min_value) / (max_value - min_value)
            for parameter in parameter_list
        ]
        if max_value != min_value
        else [zero_divide_value] * len(parameter_list)
    )


def zero_divide(numerator, denominator):
    return 0 if denominator == 0 else (numerator / denominator)
