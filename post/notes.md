## Notes

#### Convergence Rate

**Claim**: Suppose $\mathbf{M}$ is a $p \times p$ symmetric matrix of rank $r \le p$, and let $\mathbf{M}=\mathbf{U}\Lambda\mathbf{U}^{\prime}$ be its spectral decomposition, where $\mathbf{U}$ is a $p \times r$ ortho-normal matrix and $\Lambda$ is a $r \times r$ diagonal matrix. Suppose $\mathbf{\hat{M}}_n$ is a sequence of random matrices and $\{a_n\}$ is a sequence of diverging positive numbers such that  $$a_n\text{vec}(\mathbf{\hat{M}}_n - \mathbf{M}) \Rightarrow N(0,\Theta).$$ Let $\mathbf{\hat{M}}_n = \mathbf{\hat{U}}_n \hat{\Lambda}_n\mathbf{\hat{U}}_n^{\prime}$ be spectral decomposition of $\mathbf{\hat{M}}_n$. Then  $\hat{\mathbf{U}} - \mathbf{U} = O_p(1/a_n)$ and $\mathbf{U}^{\prime}(\hat{\mathbf{U}} - \mathbf{U}) + (\hat{\mathbf{U}} - \mathbf{U})^{\prime}\mathbf{U} = o_p(1/a_n)$.

**Proof**: $\hat{\mathbf{U}} - \mathbf{U} = O_p(1/a_n)$ is obvious. The claim follows by

$$\mathbf{U}^{\prime}(\hat{\mathbf{U}} - \mathbf{U}) + (\hat{\mathbf{U}} - \mathbf{U})^{\prime}\mathbf{U} = -(\hat{\mathbf{U}} - \mathbf{U})^{\prime}(\hat{\mathbf{U}} - \mathbf{U}) + \hat{\mathbf{U}}^{\prime}\hat{\mathbf{U}} - \mathbf{U}^{\prime}\mathbf{U},$$

since $\hat{\mathbf{U}}$ and $\mathbf{U}$ are both ortho-normal matrices.
