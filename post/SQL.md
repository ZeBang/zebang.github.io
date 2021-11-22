### SQL Summary

#### 1 window function

Often used when there are some conditions in groups, e.g. Top N problem. Median problem. Consecutive Problem

problem： # 185. 512, 569, 571, 180, 550, 534

**Table: Window Functions**

| Name                                                         | Description                                                  |
| :----------------------------------------------------------- | :----------------------------------------------------------- |
| [`CUME_DIST()`](https://dev.mysql.com/doc/refman/8.0/en/window-function-descriptions.html#function_cume-dist) | Cumulative distribution value                                |
| **[`DENSE_RANK()`](https://dev.mysql.com/doc/refman/8.0/en/window-function-descriptions.html#function_dense-rank)** | Rank of current row within its partition, without gaps       |
| [`FIRST_VALUE()`](https://dev.mysql.com/doc/refman/8.0/en/window-function-descriptions.html#function_first-value) | Value of argument from first row of window frame             |
| **[`LAG()`](https://dev.mysql.com/doc/refman/8.0/en/window-function-descriptions.html#function_lag)** | Value of argument from row lagging current row within partition |
| [`LAST_VALUE()`](https://dev.mysql.com/doc/refman/8.0/en/window-function-descriptions.html#function_last-value) | Value of argument from last row of window frame              |
| **[`LEAD()`](https://dev.mysql.com/doc/refman/8.0/en/window-function-descriptions.html#function_lead)** | Value of argument from row leading current row within partition |
| [`NTH_VALUE()`](https://dev.mysql.com/doc/refman/8.0/en/window-function-descriptions.html#function_nth-value) | Value of argument from N-th row of window frame              |
| [`NTILE()`](https://dev.mysql.com/doc/refman/8.0/en/window-function-descriptions.html#function_ntile) | Bucket number of current row within its partition.           |
| [`PERCENT_RANK()`](https://dev.mysql.com/doc/refman/8.0/en/window-function-descriptions.html#function_percent-rank) | Percentage rank value                                        |
| **[`RANK()`](https://dev.mysql.com/doc/refman/8.0/en/window-function-descriptions.html#function_rank)** | Rank of current row within its partition, with gaps          |
| **[`ROW_NUMBER()`](https://dev.mysql.com/doc/refman/8.0/en/window-function-descriptions.html#function_row-number)** | Number of current row within its partition                   |

template

```mysql
with t as 
(
    select 
        ..., ...,
        f() over(partition by ... order by ...) as rn
    from 
        table
)

select
    ..., ...
from
	t
where 
    f(rn)
```

#1112 rank problem

#1070 min problem

#1076 max problem. group by 不能写在窗口函数的partition by里面，因为这里涉及到两步运算，先分组计数再排序，放到partition by则变成先分组排序，逻辑错误. 这道题也可以用all，类似#1082

#### 2 self join

template

```mysql
select 
	...
from 
	table t1 join on table t2 on t1.id1 = t2.id2
where
	...
```

problem： # 603, 534, 613, 

#612 Cartesian Product

#608 join and case

#### 3 outer join

#1132 left join using(id) 保证了表1的id必在表2的id之中，这是大多数join的作用

#1126 需要和一个统计量进行比较，先计算统计量然后join到原表

#1098 易错. 搞清楚join on x where y 中x,y的位置区别. where往往省略掉left join中产生的null. 所以x不能在where之后而只能在join on 之后

problem: # 607, 577, 580

#### 4 sub table

with 之间嵌套使用

#597 two nums divided

#1174 two nums divided

#550 two nums divided while numerator has conditions

#1173

#602 two nums added

#1132 

#1164 pair的筛选两种方法：一，where(x, y) in (select ... from ...). 二，with 之间嵌套使用, 常见于先排序再选出top再输出

Other: #619

#### 5 Case

#610 列转行

#1083 列传行



#### 6 Recursive

#1613 standard recursive

#### 7  simple group by calculations

#1393 simple group by and sum(if())

#1193

#### 8 special functions

#1193 date_format(date, '%Y-%m')





求average:

#1142



累加：要么自联，要么变量

#1204

#534





references

https://zhuanlan.zhihu.com/p/341433683

https://zhuanlan.zhihu.com/p/338321823