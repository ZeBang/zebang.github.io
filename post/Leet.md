#### 1. Two Sum

Algorithm: Hash-Map.  
Time: O(n)  
Space: O(n)  
Description: Look up in hash table should be amortized O(1) time as long as the hash function was chosen carefully.


```python
def twoSum(self, nums, target):
    h = {}
    for i, num in enumerate(nums):
        complement = target - num
        if complement not in h:
            h[num] = i
        else:
            return [h[complement], i]
```

#### 11. Container With Most Water

Algorithm: Two-Pointer.  
Time: O(n)  
Space: O(1)  
Description: Move longer line will not increase the volumn. So we start from most left and right then move shorter line towards middle.


```python
def maxArea(self, height: List[int]) -> int:
    left, right = 0, len(height) - 1
    ans = 0
    while left < right:
        water = min(height[left], height[right]) * (right - left)
        ans = max(ans, water)
        if height[left] <= height[right]:
            left += 1
        else:
            right -= 1
    return ans
```

#### 15. 3Sum

Algorithm: Two-Pointer.  
Time: O(n^2)  
Space: O(1)  
Description: Sort and Two-Pointer.


```python
def threeSum(nums):
    n = len(nums)
    nums.sort()
    ans = list()
    for first in range(n - 2): # 取 n -2 防止指针溢出
        if nums[first] > 0:
            return ans
        if first > 0 and nums[first] == nums[first - 1]:
            continue
        second = first + 1
        third = n - 1
        target = -nums[first]
        while second < third:
            if second > first + 1 and nums[second] == nums[second - 1]:
                second += 1
                continue
            if second > n - 1:
                break
            if nums[second] + nums[third] > target:
                third -= 1
            elif nums[second] + nums[third] < target:
                second += 1
            else:
                ans.append([nums[first], nums[second], nums[third]])
                third -= 1
                second += 1
    return ans
```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```

* https://books.halfrost.com/leetcode/

