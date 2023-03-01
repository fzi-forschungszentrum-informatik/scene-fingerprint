"""Exemplary metric to show usage of metric handle"""


class DummyMetric:
    """Exemplary metric to show usage of metric handle
        :param first_param: first value to be calculated
        :param second_param: second value to be calculated
    """
    _value = None
    _first_param = None
    _second_param = None

    def __init__(self, first_param, second_param):
        self._first_param = first_param
        self._second_param = second_param
        print("DummyMetric init")

    @property
    def value(self):
        """Criticality value of this metric"""
        return self._value

    def calculate(self):
        """Calculated the criticality value based on """
        self._value = self._first_param - self._second_param
