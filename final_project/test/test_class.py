class test_class:
    """
    Testing out making a class

    """
    def __init__(self):
        self.outStr = "hello world class"


    def hello_world_doc(first, second, third):
        """
        Some description text

        :param second:
        :type second:
        :param third:
        :type third:
        :return: first + second
        :rtype: float
        :raises DimensionError: check dimension of matrices

        """
        return 1+1


    def hello_world(arg1, arg2, arg3):
        """
        Testing hello word

        :param arg1: the first value
        :param arg2: the first value
        :param arg3: the first value
        :type arg1: int, float,...
        :type arg2: int, float,...
        :type arg3: int, float,...
        :return: arg1/arg2 +arg3
        .. todo:: check that arg2 is non zero.
        """

        print(self.outStr)
        return 1+1.5

    def hello_world2(self):
        """
        Testing hello word2

        :returns: nothing
        """

        print(self.outStr + " 2")


