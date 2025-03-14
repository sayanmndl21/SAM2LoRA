def update_moving_average(m_m_p: float, m_c: float, n_p: float, n_c: float) -> float:
    """
    Update the moving average using a weighted sum.

    Args:
        m_m_p (float): The previous moving average.
        m_c (float): The current measurement.
        n_p (float): The weight (or count) associated with the previous average.
        n_c (float): The weight (or count) associated with the current measurement.

    Returns:
        float: The updated moving average.
    """
    m_m_n = (n_p * m_m_p + n_c * m_c) / (n_p + n_c)
    return m_m_n
