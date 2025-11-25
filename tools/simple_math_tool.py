
def is_greater(int_a: int, int_b: int) -> str:
    """
    Compares two integers and determines if the first is greater than the second.

    Args:
        int_a: The first integer to compare
        int_b: The second integer to compare

    Returns:
        A string indicating whether int_a is greater than int_b
    """
    result = int_a > int_b
    return f"{int_a} is {'greater' if result else 'not greater'} than {int_b}"
