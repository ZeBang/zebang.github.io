---
title: 'Some Notes About Python'
date: 2020-09-11
permalink: /learning/python
tags:
  - learning
---

#### How to see source code
There is a module called "inspect"
```
def foo(arg):             
    return arg
inspect.getsource(foo)
```
Or easier way use,
```
foo??
```
You will see
```
Signature: foo(arg)
Docstring: <no docstring>
Source:   
def foo(arg):                 return arg
File:      c:\users\zeban\<ipython-input-26-d60020dc647a>
Type:      function
```
You can not view python source code for `built-in function or class`, for example, `min`, `max`, `set`, etc.
The `built-in function` type is always implemented in C.
The code for this function comes from the `bltinmodule.c` source file; the [`builtin_min()`  function](http://hg.python.org/cpython/file/cd95f1276360/Python/bltinmodule.c#l1448) delegates to the [`min_max()`  utility](http://hg.python.org/cpython/file/cd95f1276360/Python/bltinmodule.c#l1345) in the same source file.

How to know whether object is a builtin function?
```
import types
types.BuiltinFunctionType
isinstance(min, types.BuiltinFunctionType)
> True
```
Or just use `help()`

References:
[RUNOOB](https://www.runoob.com/python3/python3-tutorial.html)
[Liaoxuefeng](https://www.liaoxuefeng.com/wiki/1016959663602400)
[Python official Doc](https://docs.python.org/zh-cn/3/tutorial/index.html)
[StackOverflow](https://stackoverflow.com/)
