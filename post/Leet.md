Table of contents

- [1. Two Sum](#1-two-sum)
- [11. Container With Most Water](#11-container-with-most-water)
- [15. 3Sum](#15-3sum)
- [16. 3Sum Closeat](#16-3sum-closeat)
- [18. 4Sum](#18-4sum)

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

#### 16. 3Sum Closeat

Algorithm: Two-Pointer.  
Time: O(n^2)  
Space: O(n) or O(logn) depends on which sort algorithm been used.  
Description: Now we just return smallest diff.  

Algorithm: Binary Search.  
Time: O(n^2 logn)  
Space: O(n) or O(logn) depends on which sort algorithm been used.  
Description: bisect_right().


```python
# Two Pointer
def threeSumClosest(nums, target):
    diff = float('inf')
    nums.sort()
    for i in range(len(nums)):
        lo, hi = i + 1, len(nums) - 1
        while (lo < hi):
            sum = nums[i] + nums[lo] + nums[hi]
            if abs(target - sum) < abs(diff):
                diff = target - sum
            if sum < target:
                lo += 1
            else:
                hi -= 1
        if diff == 0:
            break
    return target - diff
```


```python
# Binary Search
def threeSumCloseat(nums, target):
    diff = float('inf')
    nums.sort()
    for i in range(len(nums)):
        for j in range(i + 1, len(nums)):
            complement = target - nums[i] - nums[j]
            hi = bisect_right(nums, complement, j + 1)
            lo = hi - 1
            if hi < len(nums) and abs(complement - nums[hi]) < abs(diff):
                diff = complement - nums[hi]
            if lo > j and abs(complement - nums[lo]) < abs(diff):
                diff = complement - nums[lo]
        if diff == 0:
            break
    return target - diff
```

#### 18. 4Sum

Algorithm: Two-Pointer.  
Time: O(n^3)  
Space: O(n)

Algorithm: Hash-Map.  
Time: O(n^3)  
Space: O(n)


```python
# Two Pointer
def fourSum(nums, target):
    def kSum(nums, target, k):
        res = []
        if len(nums) == 0 or nums[0] * k > target or nums[-1] * k < target:
            return res
        if k == 2:
            return twoSum(nums, target)
        for i in range(len(nums)):
            if i == 0 or nums[i - 1] != nums[i]: # no duplicate nums[i]
                 for set in kSum(nums[i + 1:], target - nums[i], k - 1):
                    res.append([nums[i]] + set)
        return res

    def twoSum(nums, target):
            res = []
            lo, hi = 0, len(nums) - 1
            while (lo < hi):
                sum = nums[lo] + nums[hi]
                if sum < target or (lo > 0 and nums[lo] == nums[lo - 1]):
                    lo += 1
                elif sum > target or (hi < len(nums) - 1 and nums[hi] == nums[hi + 1]):
                    hi -= 1
                else:
                    res.append([nums[lo], nums[hi]])
                    lo += 1
                    hi -= 1
            return res

    nums.sort()
    return kSum(nums, target, 4)
```


```python
# Hash Map
def fourSum(nums, target):
    def kSum(nums, target, k):
        if len(nums) == 0 or nums[0] * k > target or nums[-1] * k < target:
            return []
        if k == 2:
            return twoSum(nums, target)
        res = []
        for i in range(len(nums)):
            if i == 0 or nums[i - 1] != nums[i]: # no duplicate nums[i]
                 for set in kSum(nums[i + 1:], target - nums[i], k - 1):
                    res.append([nums[i]] + set)
        return res

    def twoSum(nums, target):
        res = []
        s = set()
        for i in range(len(nums)):
            if len(res) == 0 or res[-1][1] != nums[i]:
                if target - nums[i] in s:
                    res.append([target - nums[i], nums[i]])
            s.add(nums[i])
        return res

    nums.sort()
    return kSum(nums, target, 4)
```

* https://books.halfrost.com/leetcode/

