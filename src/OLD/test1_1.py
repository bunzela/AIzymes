def test_function_1(main_variable):
    print("test_1")

def test_function_2_executed_from_1():
    print("test_function_2_executed_from_1")
    test_function_2()  # This should call test_function_2