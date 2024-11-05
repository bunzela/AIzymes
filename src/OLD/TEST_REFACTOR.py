import runpy

MODULES = ['test_2', 'test_1']
for module in MODULES:
    module_globals = runpy.run_path(f'src/{module}.py')
    for func_name, func in module_globals.items():
        if callable(func):
            globals()[func_name] = func