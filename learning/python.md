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

## Mutable, Immutable and Pointer
```
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
l1 = ListNode(1)
l2 = ListNode(2)
dummy = cur = ListNode(0)
cur.next = l2

dummy = cur = [1,2]
cur[1] = 3
```
see [How to make an immutable object in python](https://stackoverflow.com/questions/4828080/how-to-make-an-immutable-object-in-python)

##  [`itertools`](https://docs.python.org/3/library/itertools.html#module-itertools "itertools: Functions creating iterators for efficient looping.")  — Functions creating iterators for efficient looping[¶](https://docs.python.org/3/library/itertools.html#module-itertools "Permalink to this headline")
see LeetCode example [38. Count and Say](https://leetcode.com/problems/count-and-say/discuss/15999/4-5-lines-Python-solutions) using "groupby".



## References:
[RUNOOB](https://www.runoob.com/python3/python3-tutorial.html)
[Liaoxuefeng](https://www.liaoxuefeng.com/wiki/1016959663602400)
[Python official Doc](https://docs.python.org/zh-cn/3/tutorial/index.html)
[StackOverflow](https://stackoverflow.com/)
[awesome-python](https://awesome-python.com/)
