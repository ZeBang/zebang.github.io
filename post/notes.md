## Notes on Functional Analysis

张恭庆《泛函分析讲义》是一本好书，这里是我的学习笔记，包括一些章节的梳理，细节的补充，纰漏的说明。

### Chapter 1 Metric Space

* **metric space** is a set with distance.

* **distance** is a real-valued bi-variate function.

* The reason to define distance is to describe **convergence**.

* The concept of **close** set and **complete**(all Cauchy series converge) can be introduced to metric space.

  * Convergence series are Cauchy but converse is not true, 

    * e.g. 有理数在通常定义的距离意义下不是完备的 (有理数空间下的柯西列有可能收敛于无理数)

    * e.g. 实数是完备的，标准的实数构造包含有理数的柯西列

    * e.g. The concept of complete is influenced by the distance:

      $\exist$ two metric space $(\mathcal{X}, d_1), (\mathcal{X}, d_2)$ defined on same set $\mathcal{X}$ but different distance $d_1, d_2$, such that one is complete but the other is not

      $\mathcal{X}=C[0,1]$, $d_1 = \max_{0 \leq t \leq 1}|x(t)-y(t)|$, $d_2 = \int_0^1|x(t)-y(t)|dt$.

  * Close is a concept for subset in topolycal space, i.e. complement of open set. But in metric space, close set is defined by convergence
  
  * Close subset of complete space is complete; Complete subset of metric space is close.
  
* Given two metric space, we can define **continuous mapping**, **contraction mapping**.

* One important theorem related to contraction mapping is **Banach fixed-point theorem**.

  

  **Example 1.1.7** Uniform convergence preserves the continuity of continuous functions。

  **Definition 1.1.10** Contraction mapping must be continues mapping.





---

**Reference**

汪林, "泛函分析中的反例". 高等教育出版社.
