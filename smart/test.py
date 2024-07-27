from RestrictedPython import compile_restricted

byte_code = compile_restricted(
    "2 + 2",
    filename='<inline code>',
    mode='eval'
)
eval(byte_code)