



## leetCode in Python



[toc]



### Template

Sort List 148

Merge List 143

Vec List 143

Reverse List 206

Circle List 142

Index List 19

Monotone Queue 239



### Binary Search

Standard binary search: 704 - 35 - 34 - 69 - 367

#### 704. Binary Search

Given an array of integers `nums` which is sorted in ascending order, and an integer `target`, write a function to search `target` in `nums`. If `target` exists, then return its index. Otherwise, return `-1`.

You must write an algorithm with `O(log n)` runtime complexity.

 

**Example 1:**

```
Input: nums = [-1,0,3,5,9,12], target = 9
Output: 4
Explanation: 9 exists in nums and its index is 4
```

```python
class Solution:
    def search(self, nums: List[int], target: int) -> int:
        left, right = 0, len(nums) - 1
        while left <= right:
            middle = (left + right) // 2
            if nums[middle] < target:
                left = middle + 1
            elif nums[middle] > target:
                right = middle - 1
            else:
                return middle
        return -1
```

```python
class Solution:
    def search(self, nums: List[int], target: int) -> int:
        left, right  = 0, len(nums)
        while left < right:
            mid = (left + right) // 2
            if nums[mid] < target:
                left = mid + 1
            elif nums[mid] > target:
                right = mid
            else:
                return mid
        return -1

```

#### 35. Search Insert Position

Given a sorted array of distinct integers and a target value, return the index if the target is found. If not, return the index where it would be if it were inserted in order.

You must write an algorithm with `O(log n)` runtime complexity.

 

**Example 1:**

```
Input: nums = [1,3,5,6], target = 5
Output: 2
```

```python
class Solution:
    def searchInsert(self, nums: List[int], target: int) -> int:
        left, right = 0, len(nums) - 1
        while left <= right:
            mid = left + (right - left) // 2 # mid = (right + left) // 2
            if nums[mid] < target:
                left = mid + 1
            elif nums[mid] > target:
                right = right - 1
            else:
                return mid
        return left
```

#### 34. Find First and Last Position of Element in Sorted Array

Given an array of integers `nums` sorted in ascending order, find the starting and ending position of a given `target` value.

If `target` is not found in the array, return `[-1, -1]`.

You must write an algorithm with `O(log n)` runtime complexity.

 

**Example 1:**

```
Input: nums = [5,7,7,8,8,10], target = 8
Output: [3,4]
```

```python
class Solution:
    def searchRange(self, nums: List[int], target: int) -> List[int]:
        if len(nums) == 0:
            return [-1, -1]
        if nums[0] > target:
            return [-1, -1]
        
        left, right = 0, len(nums) - 1 # 注意len - 1
        
        left = self.find_first(nums, target, left, right)
        right = self.find_end(nums, target, left, right)
        
        return [left, right]
    
    def find_first(self, nums, target, left, right):
        while left <= right:
            mid = left + (right - left) // 2
            if nums[mid] < target:
                left = mid + 1
            else:
                right = mid - 1
        if left < len(nums) and nums[left] == target:
            return left
        if right < len(nums) and nums[right] == target:
            return right
        return -1
    
    def find_end(self, nums, target, left, right):
        while left <= right:
            mid = left + (right - left) // 2
            if nums[mid] > target:
                right = mid - 1
            else:
                left = mid + 1
        if right < len(nums) and nums[right] == target:
            return right
        if left < len(nums) and nums[left] == target:
            return left
        return -1
```

```python
class Solution:
    def searchRange(self, nums: List[int], target: int) -> List[int]:
        if len(nums) == 0:
            return [-1, -1]
        if nums[0] > target:
            return [-1, -1]
        
        left, right = 0, len(nums) - 1
        
        left = self.find_first(nums, target, left, right)
        right = self.find_end(nums, target, left, right)
        
        return [left, right]
    
    def find_first(self, nums, target, left, right):
        while left + 1 < right:
            mid = left + (right - left) // 2
            if nums[mid] < target:
                left = mid
            else:
                right = mid
        if left < len(nums) and nums[left] == target:
            return left
        if right < len(nums) and nums[right] == target:
            return right
        return -1
    
    def find_end(self, nums, target, left, right):
        while left + 1 < right:
            mid = left + (right - left) // 2
            if nums[mid] > target:
                right = mid
            else:
                left = mid
        if right < len(nums) and nums[right] == target:
            return right
        if left < len(nums) and nums[left] == target:
            return left
        return -1
```

```python
class Solution:
    def searchRange(self, nums: List[int], target: int) -> List[int]:
        if len(nums) == 0:
            return [-1, -1]
        if nums[0] > target:
            return [-1, -1]
        
        left, right = 0, len(nums)
        
        left = self.find_first(nums, target, left, right)
        right = self.find_end(nums, target, left, right)
        
        return [left, right]
    
    def find_first(self, nums, target, left, right):
        while left < right:
            mid = left + (right - left) // 2
            if nums[mid] < target:
                left = mid + 1
            else:
                right = mid
        if left < len(nums) and nums[left] == target:
            return left
        if right < len(nums) and nums[right] == target:
            return right
        return -1
    
    def find_end(self, nums, target, left, right):
        while left < right:
            mid = left + (right - left) // 2
            if nums[mid] > target:
                right = mid
            else:
                left = mid + 1
        if right < len(nums) and nums[right] == target:
            return right
        if left - 1 < len(nums) and nums[left - 1] == target:
            return left - 1
        return -1
```

```python
class Solution:
    def searchRange(self, nums: List[int], target: int) -> List[int]:
        if len(nums) == 0:
            return [-1, -1]
        if nums[0] > target:
            return [-1, -1]
        
        left = bisect_left(nums, target, 0, len(nums) - 1)
        right = bisect_right(nums, target, 0, len(nums) - 1)
        
        if nums[left] == target:
            if nums[right] == target:
                return [left, right]
            else:
                return [left, right - 1]
        return [-1, -1]
```

#### 69. Sqrt(x)

Given a non-negative integer `x`, compute and return *the square root of* `x`.

Since the return type is an integer, the decimal digits are **truncated**, and only **the integer part** of the result is returned.

**Note:** You are not allowed to use any built-in exponent function or operator, such as `pow(x, 0.5)` or `x ** 0.5`.



**Example 1:**

```
Input: x = 4
Output: 2
```

```python
class Solution:
    def mySqrt(self, x: int) -> int:
        if x < 2: return x
        start = 1
        end = x // 2
        while start <= end:
            mid = start + (end - start) // 2
            if mid*mid > x:
                end = mid - 1
            else:
                start = mid + 1
        return start - 1
```

#### 367. Valid Perfect Square

Given a **positive** integer *num*, write a function which returns True if *num* is a perfect square else False.

**Follow up:** **Do not** use any built-in library function such as `sqrt`.



**Example 1:**

```
Input: num = 16
Output: true
```

```python
class Solution:
    def isPerfectSquare(self, num: int) -> bool:
        start = 1
        end = num
        while start <= end:
            mid = start + (end - start) // 2
            if mid*mid == num:
                return True
            if mid*mid < num:
                start = mid + 1
            else:
                end = mid - 1
        return False
```





#### 33. Search in Rotated Sorted Array

There is an integer array `nums` sorted in ascending order (with **distinct** values).

Prior to being passed to your function, `nums` is **rotated** at an unknown pivot index `k` (`0 <= k < nums.length`) such that the resulting array is `[nums[k], nums[k+1], ..., nums[n-1], nums[0], nums[1], ..., nums[k-1]]` (**0-indexed**). For example, `[0,1,2,4,5,6,7]` might be rotated at pivot index `3` and become `[4,5,6,7,0,1,2]`.

Given the array `nums` **after** the rotation and an integer `target`, return *the index of* `target` *if it is in* `nums`*, or* `-1` *if it is not in* `nums`.

You must write an algorithm with `O(log n)` runtime complexity.

 

**Example 1:**

```
Input: nums = [4,5,6,7,0,1,2], target = 0
Output: 4
```

```python
# Approach 1: Binary search
class Solution:
    def search(self, nums: List[int], target: int) -> int:
 		def rotate_index_search(start, end, nums):
            while start + 1 < end:
                mid = start + (end - start) // 2
                if nums[mid] > nums[end]:
                    start = mid
                else:
                    end = mid
            if nums[start] > nums[end]:
                return end
            else:
                return start
        
        def bisearch(start, end, nums, target):
            while start + 1 < end:
                mid = start + (end - start) // 2
                if nums[mid] < target:
                    start = mid
                else:
                    end = mid
            if nums[start] == target:
                return start
            if nums[end] == target:
                return end
            return -1
        
        start = 0
        end = len(nums) - 1
        rotate_index = rotate_index_search(start, end, nums)
        
        if rotate_index == 0:
            return bisearch(start, end, nums, target)
        if target >= nums[0]:
            return bisearch(start, len(nums[0:rotate_index]) - 1, nums[0:rotate_index], target)
        else:
            ans = bisearch(start, len(nums[rotate_index:]) - 1 , nums[rotate_index:], target)
            return -1 if ans == -1 else rotate_index + ans
```

```python
# Approach 2: One-pass Binary Search
class Solution:
    def search(self, nums: List[int], target: int) -> int:
        start, end = 0, len(nums) - 1
        while start <= end:
            mid = start + (end - start) // 2
            if nums[mid] == target:
                return mid
            elif nums[mid] >= nums[start]:
                if target >= nums[start] and target < nums[mid]:
                    end = mid - 1
                else:
                    start = mid + 1
            else:
                if target <= nums[end] and target > nums[mid]:
                    start = mid + 1
                else:
                    end = mid - 1
        return -1
```













#### 81. Search in Rotated Sorted Array II

There is an integer array `nums` sorted in non-decreasing order (not necessarily with **distinct** values).

Before being passed to your function, `nums` is **rotated** at an unknown pivot index `k` (`0 <= k < nums.length`) such that the resulting array is `[nums[k], nums[k+1], ..., nums[n-1], nums[0], nums[1], ..., nums[k-1]]` (**0-indexed**). For example, `[0,1,2,4,4,4,5,6,6,7]` might be rotated at pivot index `5` and become `[4,5,6,6,7,0,1,2,4,4]`.

Given the array `nums` **after** the rotation and an integer `target`, return `true` *if* `target` *is in* `nums`*, or* `false` *if it is not in* `nums`*.*

You must decrease the overall operation steps as much as possible.

 

**Example 1:**

```
Input: nums = [2,5,6,0,0,1,2], target = 0
Output: true
```

```python
class Solution:
    def search(self, nums: List[int], target: int) -> int:
        start, end = 0, len(nums) - 1
        while start <= end:
            mid = start + (end - start) // 2
            if nums[mid] == target:
                return True
            if nums[start] == nums[mid] and nums[mid] == nums[end]: # only exception: nums[start] == nums[mid] == nums[end]
                start += 1
                end -= 1
            elif nums[mid] >= nums[start]:
                if target >= nums[start] and target < nums[mid]:
                    end = mid - 1
                else:
                    start = mid + 1
            else:
                if target <= nums[end] and target > nums[mid]:
                    start = mid + 1
                else:
                    end = mid - 1
        return False
```



#### 153. Find Minimum in Rotated Sorted Array

Suppose an array of length `n` sorted in ascending order is **rotated** between `1` and `n` times. For example, the array `nums = [0,1,2,4,5,6,7]` might become:

- `[4,5,6,7,0,1,2]` if it was rotated `4` times.
- `[0,1,2,4,5,6,7]` if it was rotated `7` times.

Notice that **rotating** an array `[a[0], a[1], a[2], ..., a[n-1]]` 1 time results in the array `[a[n-1], a[0], a[1], a[2], ..., a[n-2]]`.

Given the sorted rotated array `nums` of **unique** elements, return *the minimum element of this array*.

You must write an algorithm that runs in `O(log n) time.`

 

**Example 1:**

```
Input: nums = [3,4,5,1,2]
Output: 1
Explanation: The original array was [1,2,3,4,5] rotated 3 times.
```

```python
class Solution:
    def findMin(self, nums: List[int]) -> int:
        start = 0
        end = len(nums) - 1
        while start + 1 < end:
            mid = start + (end - start) // 2
            if nums[mid] > nums[end]:
                start = mid
            else:
                end = mid
        return min(nums[start], nums[end])

```

```python
class Solution:
    def findMin(self, nums: List[int]) -> int:
        start = 0
        end = len(nums) - 1
        while start <= end:
            mid = (end + start) // 2
            if nums[mid] > nums[start - 1]:
                start = mid + 1
            else:
                end = mid - 1
        return min(nums[start], nums[end])
```

#### 74. Search a 2D Matrix

Write an efficient algorithm that searches for a value in an `m x n` matrix. This matrix has the following properties:

- Integers in each row are sorted from left to right.
- The first integer of each row is greater than the last integer of the previous row.

 

**Example 1:**

![img](https://assets.leetcode.com/uploads/2020/10/05/mat.jpg)

```
Input: matrix = [[1,3,5,7],[10,11,16,20],[23,30,34,60]], target = 3
Output: true
```

```python
# two bisect
class Solution(object):
    def searchMatrix(self, matrix, target):
        M, N = len(matrix), len(matrix[0])
        col0 = [row[0] for row in matrix]
        target_row = bisect.bisect_right(col0, target) - 1
        if target_row < 0:
            return False
        target_col = bisect.bisect_left(matrix[target_row], target)
        if target_col >= N:
            return False
        if matrix[target_row][target_col] == target:
            return True
        return False
```

```python
# Global bisect
class Solution(object):
    def searchMatrix(self, matrix, target):
        M, N = len(matrix), len(matrix[0])
        left, right = 0, M * N - 1
        while left <= right:
            mid = left + (right - left) // 2
            cur = matrix[mid // N][mid % N]
            if cur == target:
                return True
            elif cur < target:
                left = mid + 1
            else:
                right = mid - 1
        return False
```



#### 162. Find Peak Element

A peak element is an element that is strictly greater than its neighbors.

Given an integer array `nums`, find a peak element, and return its index. If the array contains multiple peaks, return the index to **any of the peaks**.

You may imagine that `nums[-1] = nums[n] = -∞`.

You must write an algorithm that runs in `O(log n)` time.

 

**Example 1:**

```
Input: nums = [1,2,3,1]
Output: 2
Explanation: 3 is a peak element and your function should return the index number 2.
```

```python
class Solution:
    def findPeakElement(self, nums: List[int]) -> int:
        left = 0
        right = len(nums) - 1
        while left < right:
            mid = left + (right - left)//2
            if nums[mid] < nums[mid+1]:
                left = mid + 1
            else:
                right = mid
        return left
```





#### 278. First Bad Version

You are a product manager and currently leading a team to develop a new product. Unfortunately, the latest version of your product fails the quality check. Since each version is developed based on the previous version, all the versions after a bad version are also bad.

Suppose you have `n` versions `[1, 2, ..., n]` and you want to find out the first bad one, which causes all the following ones to be bad.

You are given an API `bool isBadVersion(version)` which returns whether `version` is bad. Implement a function to find the first bad version. You should minimize the number of calls to the API.

 

**Example 1:**

```
Input: n = 5, bad = 4
Output: 4
Explanation:
call isBadVersion(3) -> false
call isBadVersion(5) -> true
call isBadVersion(4) -> true
Then 4 is the first bad version.
```

```python
# The isBadVersion API is already defined for you.
# @param version, an integer
# @return an integer
# def isBadVersion(version):

class Solution:
    def firstBadVersion(self, n):
        start = 1
        end = n
        while start + 1 < end:
            mid = start + (end - start) // 2
            if not isBadVersion(mid):
                start = mid + 1
            else:
                end = mid - 1
        if isBadVersion(start):
            return start
        elif isBadVersion(end):
            return end
        else:
            return end + 1
```

```python
class Solution:
    def firstBadVersion(self, n):
        start = 1
        end = n
        while start <= end:
            mid = (end + start) // 2
            if not isBadVersion(mid):
                start = mid + 1
            else:
                end = mid - 1
        if isBadVersion(start):
            return start
        else:
            return end
```

```python
class Solution:
    def firstBadVersion(self, n):
        start = 0
        end = n
        while start <= end:
            mid = (start + end) // 2
            if isBadVersion(mid):
                end = mid - 1
            else:
                start = mid + 1
        if isBadVersion(end):
            return end
        else:
            return end + 1
```



#### 4. Median of Two Sorted Arrays

Given two sorted arrays `nums1` and `nums2` of size `m` and `n` respectively, return **the median** of the two sorted arrays.

The overall run time complexity should be `O(log (m+n))`.

 

**Example 1:**

```
Input: nums1 = [1,3], nums2 = [2]
Output: 2.00000
Explanation: merged array = [1,2,3] and median is 2.
```

**Example 2:**

```
Input: nums1 = [1,2], nums2 = [3,4]
Output: 2.50000
Explanation: merged array = [1,2,3,4] and median is (2 + 3) / 2 = 2.5.
```

```python
class Solution:
    def findMedianSortedArrays(self, nums1: List[int], nums2: List[int]) -> float:
        if len(nums1) > len(nums2):
            return self.findMedianSortedArrays(nums2, nums1)

        infinty = float('inf')
        m, n = len(nums1), len(nums2)
        left, right = 0, m
        # median1：前一部分的最大值
        # median2：后一部分的最小值
        median1, median2 = 0, 0

        while left <= right:
            # 前一部分包含 nums1[0 .. i-1] 和 nums2[0 .. j-1]
            # 后一部分包含 nums1[i .. m-1] 和 nums2[j .. n-1]
            i = left + (right - left) // 2
            j = (m + n + 1) // 2 - i
            
            # nums_im1, nums_i, nums_jm1, nums_j 分别表示 nums1[i-1], nums1[i], nums2[j-1], nums2[j]
            nums_im1 = (-infinty if i == 0 else nums1[i - 1])
            nums_i = (infinty if i == m else nums1[i])
            nums_jm1 = (-infinty if j == 0 else nums2[j - 1])
            nums_j = (infinty if j == n else nums2[j])

            if nums_im1 > nums_j:
                right = i - 1
            else:
                left = i + 1
                median1, median2 = max(nums_im1, nums_jm1), min(nums_i, nums_j)

        return (median1 + median2) / 2 if (m + n) % 2 == 0 else median1
```



```python
class Solution:
    def findMedianSortedArrays(self, nums1: List[int], nums2: List[int]) -> float:
        
        if len(nums1) > len(nums2):
            nums1, nums2 = nums2, nums1

        m, n = len(nums1), len(nums2)
        totalLeft = (m + n + 1) // 2
        left = 0
        right = m

        while (left < right):
            i = left + (right - left + 1) // 2
            j = totalLeft - i
            if nums1[i - 1] > nums2[j]:
                right = i - 1
            else:
                left = i

        i = left
        j = totalLeft - i

        nums1LeftMax = -float('inf') if i==0 else nums1[i-1]
        nums1RightMin = float('inf') if i==m else nums1[i]
        nums2LeftMax = -float('inf') if j==0 else nums2[j-1]
        nums2RightMin = float('inf') if j==n else nums2[j]

        if (m + n) % 2 == 1:
            return max(nums1LeftMax, nums2LeftMax)
        else:
            return (max(nums1LeftMax, nums2LeftMax) + min(nums1RightMin, nums2RightMin))/2
```



#### 287. Find the Duplicate Number

Given an array of integers `nums` containing `n + 1` integers where each integer is in the range `[1, n]` inclusive.

There is only **one repeated number** in `nums`, return *this repeated number*.

You must solve the problem **without** modifying the array `nums` and uses only constant extra space.

 

**Example 1:**

```
Input: nums = [1,3,4,2,2]
Output: 2
```

```python
class Solution:
    def findDuplicate(self, nums: List[int]) -> int:
        slow, fast = 0, 0
        while True:
            slow, fast = nums[slow], nums[nums[fast]]
            if slow == fast:
                break
        slow = 0
        while slow != fast:
            slow, fast = nums[slow], nums[fast]
        return slow
```

```python
class Solution:
    def findDuplicate(self, nums: List[int]) -> int:
        n = len(nums)
        left = 0
        right = n - 1
        
        while left <= right:
            mid = (left + right) // 2
            count = 0
            for i in range(n):
                if nums[i] <= mid:
                    count += 1
            if count <= mid:
                left = mid + 1
            else:
                right = mid - 1
                ans = mid

        return ans
```





### Two Pointer

Slow and Fast two pointer: 26 - 27 - 283 - 844 - 977 



#### 1. Two Sum

Given an array of integers `nums` and an integer `target`, return *indices of the two numbers such that they add up to `target`*.

You may assume that each input would have ***exactly\* one solution**, and you may not use the *same* element twice.

You can return the answer in any order.

 

**Example 1:**

```
Input: nums = [2,7,11,15], target = 9
Output: [0,1]
Output: Because nums[0] + nums[1] == 9, we return [0, 1].
```




```python
# why cannot use two pointer: since not sorted (if nums.sort() then lose index)
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        h = {}
        for i, num in enumerate(nums):
            complement = target - num
            if complement not in h:
                h[num] = i
            else:
                return [h[complement], i]
```



#### 3. Longest Substring Without Repeating Characters

Given a string `s`, find the length of the **longest substring** without repeating characters.

**Example 1:**

```
Input: s = "abcabcbb"
Output: 3
Explanation: The answer is "abc", with the length of 3.
```

```python
class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        start = -1
        ans = 0
        d = {} # 哈希表，记录每个字符最近出现的位置
        for i in range(len(s)):
            if s[i] in d and d[s[i]] > start: # 最接近出现位置在start之后
                start = d[s[i]] # 需要更新start到之前重复位置
                d[s[i]] = i # 更新当前字母最新出现位置
            else:
                d[s[i]] = i
                ans = max(i - start, ans)
        return ans
```

```python
class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        occ = set() # 哈希集合，记录每个字符是否出现过
        n = len(s) # 不用len(s)-1是因为用len(s)对于len(s)=0可以直接跳过下面for循环
        rk, ans = -1, 0 # 右指针，初始值为 -1，相当于我们在字符串的左边界的左侧，还没有开始移动
        for i in range(n):
            if i != 0:
                occ.remove(s[i - 1]) # 左指针向右移动一格，移除一个字符
            while rk < n - 1 and s[rk + 1] not in occ:
                occ.add(s[rk + 1]) 
                rk += 1 # 不断地移动右指针
            ans = max(ans, rk - i + 1)  # 第 i 到 rk 个字符是一个极长的无重复字符子串
        return ans
```



#### 9. Palindrome Number

Given an integer `x`, return `true` if `x` is palindrome integer.

An integer is a **palindrome** when it reads the same backward as forward. For example, `121` is palindrome while `123` is not.

 

**Example 1:**

```
Input: x = 121
Output: true
```


```python
class Solution:
    def isPalindrome(self, x: int) -> bool:
        x = str(x) # 字符串化
        left = 0
        right = len(x) - 1
        while left <= right:
            if x[left] != x[right]:
                return False
            left += 1
            right -= 1
        return True
```

```python
class Solution:
    def isPalindrome(self, x: int) -> bool:
        result = 0
        x0 = x
        if x < 0:
           return False
        while x:
            result = result * 10 + x % 10
            x //= 10
        return True if result == x0 else False
```

```python
class Solution:
    def isPalindrome(self, x: int) -> bool:
        if x < 0:
            return False
        rev = 0
        n = x
        while n:
            n, tail = divmod(n, 10)
            rev = rev * 10 + tail
        return True if rev == x else False
```



#### 11. Container With Most Water

Given `n` non-negative integers `a1, a2, ..., an` , where each represents a point at coordinate `(i, ai)`. `n` vertical lines are drawn such that the two endpoints of the line `i` is at `(i, ai)` and `(i, 0)`. Find two lines, which, together with the x-axis forms a container, such that the container contains the most water.

**Notice** that you may not slant the container.

 

**Example 1:**

![img](https://s3-lc-upload.s3.amazonaws.com/uploads/2018/07/17/question_11.jpg)

```
Input: height = [1,8,6,2,5,4,8,3,7]
Output: 49
Explanation: The above vertical lines are represented by array [1,8,6,2,5,4,8,3,7]. In this case, the max area of water (blue section) the container can contain is 49.
```

Algorithm: Two-Pointer.  
Time: O(n)  
Space: O(1)  
Description: Move longer line will not increase the volumn. So we start from most left and right then move shorter line towards middle.


```python
class Solution:
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

Given an array `nums` of *n* integers and an integer `target`, find three integers in `nums` such that the sum is closest to `target`. Return the sum of the three integers. You may assume that each input would have exactly one solution.

 

**Example 1:**

```
Input: nums = [-1,2,1,-4], target = 1
Output: 2
Explanation: The sum that is closest to the target is 2. (-1 + 2 + 1 = 2).
```

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
    n = len(nums)
    for i in range(n):
        left, right = i + 1, n - 1
        while (left < right): # 隐含了如果i=n-1则left=n right=n-1不执行
            sum = nums[i] + nums[left] + nums[right]
            if abs(sum - target) < abs(diff):
                diff = target - sum
            if sum <= target: left += 1
            if sum > target: right -= 1
            if sum == target: break
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

Given an array `nums` of `n` integers, return *an array of all the **unique** quadruplets* `[nums[a], nums[b], nums[c], nums[d]]` such that:

- `0 <= a, b, c, d < n`
- `a`, `b`, `c`, and `d` are **distinct**.
- `nums[a] + nums[b] + nums[c] + nums[d] == target`

You may return the answer in **any order**.

 

**Example 1:**

```
Input: nums = [1,0,-1,0,-2,2], target = 0
Output: [[-2,-1,1,2],[-2,0,0,2],[-1,0,0,1]]
```

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
                 for ans in kSum(nums[i + 1:], target - nums[i], k - 1):
                    res.append([nums[i]] + ans)
        return res

    def twoSum(nums, target):
            res = []
            lo, hi = 0, len(nums) - 1
            while (lo < hi):
                sum = nums[lo] + nums[hi]
                if sum < target or (lo > 0 and nums[lo] == nums[lo - 1]): # no duplicate nums[lo]
                    lo += 1
                elif sum > target or (hi < len(nums) - 1 and nums[hi] == nums[hi + 1]): # no duplicate nums[hi]
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
                 for ans in kSum(nums[i + 1:], target - nums[i], k - 1):
                    res.append([nums[i]] + ans)
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

```python
# Hash Map
class Solution(object):
    def fourSum(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[List[int]]
        """
        # use a dict to store value:showtimes
        hashmap = dict()
        for n in nums:
            if n in hashmap:
                hashmap[n] += 1
            else: 
                hashmap[n] = 1
        
        # good thing about using python is you can use set to drop duplicates.
        ans = set()
        for i in range(len(nums)):
            for j in range(i + 1, len(nums)):
                for k in range(j + 1, len(nums)):
                    val = target - (nums[i] + nums[j] + nums[k])
                    if val in hashmap:
                        # make sure no duplicates.
                        count = (nums[i] == val) + (nums[j] == val) + (nums[k] == val)
                        if hashmap[val] > count:
                            ans.add(tuple(sorted([nums[i], nums[j], nums[k], val])))
                    else:
                        continue
        return ans
```







#### 26. Remove Duplicates from Sorted Array

Given a sorted array *nums*, remove the duplicates [**in-place**](https://en.wikipedia.org/wiki/In-place_algorithm) such that each element appears only *once* and returns the new length.

Do not allocate extra space for another array, you must do this by **modifying the input array [in-place](https://en.wikipedia.org/wiki/In-place_algorithm)** with O(1) extra memory.

**Clarification:**

Confused why the returned value is an integer but your answer is an array?

Note that the input array is passed in by **reference**, which means a modification to the input array will be known to the caller as well.

Internally you can think of this:

```
// nums is passed in by reference. (i.e., without making a copy)
int len = removeDuplicates(nums);

// any modification to nums in your function would be known by the caller.
// using the length returned by your function, it prints the first len elements.
for (int i = 0; i < len; i++) {
    print(nums[i]);
}
```



**Example 1:**

```
Input: nums = [1,1,2]
Output: 2, nums = [1,2]
Explanation: Your function should return length = 2, with the first two elements of nums being 1 and 2 respectively. It doesn't matter what you leave beyond the returned length.
```


```python
def removeDuplicates(nums):
	nums[:] = sorted(set(nums))
    return len(nums)
```


```python
def removeDuplicates(nums):    
    if len(nums) == 0 or len(nums) == 1:
        return len(nums)
    slow = 0
    for fast in range(1, len(nums)):
        if nums[fast] != nums[slow]:
            slow += 1                  # 必须先slow += 1
            nums[slow] = nums[fast]
    return slow + 1
```



#### 27. Remove Element

Given an integer array `nums` and an integer `val`, remove all occurrences of `val` in `nums` [**in-place**](https://en.wikipedia.org/wiki/In-place_algorithm). The relative order of the elements may be changed.

Since it is impossible to change the length of the array in some languages, you must instead have the result be placed in the **first part** of the array `nums`. More formally, if there are `k` elements after removing the duplicates, then the first `k` elements of `nums` should hold the final result. It does not matter what you leave beyond the first `k` elements.

Return `k` *after placing the final result in the first* `k` *slots of* `nums`.

Do **not** allocate extra space for another array. You must do this by **modifying the input array [in-place](https://en.wikipedia.org/wiki/In-place_algorithm)** with O(1) extra memory.

**Example 1:**

```
Input: nums = [3,2,2,3], val = 3
Output: 2, nums = [2,2,_,_]
Explanation: Your function should return k = 2, with the first two elements of nums being 2.
It does not matter what you leave beyond the returned k (hence they are underscores).
```

```python
class Solution:
    def removeElement(self, nums: List[int], val: int) -> int:
        index = 1
        n = len(nums)
        j = 0
        i = 0
        while j <= n - 1:
            if nums[i] == val:
                nums[i], nums[n-index] = nums[n-index], nums[i]
                index += 1
            else:
                i += 1
            j += 1
        return n - index + 1
```

```python
class Solution:
    def removeElement(self, nums: List[int], val: int) -> int:
        slow = 0
        for fast in nums:
            if fast != val:
                nums[slow] = fast
                slow += 1
        return slow
```



#### 75. Sort Colors

Given an array `nums` with `n` objects colored red, white, or blue, sort them **[in-place](https://en.wikipedia.org/wiki/In-place_algorithm)** so that objects of the same color are adjacent, with the colors in the order red, white, and blue.

We will use the integers `0`, `1`, and `2` to represent the color red, white, and blue, respectively.

You must solve this problem without using the library's sort function.

 

**Example 1:**

```
Input: nums = [2,0,2,1,1,0]
Output: [0,0,1,1,2,2]
```

```python
class Solution:
    def sortColors(self, nums: List[int]) -> None:
        """
        Dutch National Flag problem solution.
        """
        # for all idx < p0 : nums[idx < p0] = 0
        # curr is an index of element under consideration
        p0 = curr = 0
        # for all idx > p2 : nums[idx > p2] = 2
        p2 = len(nums) - 1

        while curr <= p2:
            if nums[curr] == 0:
                nums[p0], nums[curr] = nums[curr], nums[p0]
                p0 += 1
                curr += 1
            elif nums[curr] == 2:
                nums[curr], nums[p2] = nums[p2], nums[curr]
                p2 -= 1 # note no curr addition
            else:
                curr += 1
```



#### 80. Remove Duplicates from Sorted Array II

Given a sorted array *nums*, remove the duplicates [**in-place**](https://en.wikipedia.org/wiki/In-place_algorithm) such that duplicates appeared at most *twice* and return the new length.

Do not allocate extra space for another array; you must do this by **modifying the input array [in-place](https://en.wikipedia.org/wiki/In-place_algorithm)** with O(1) extra memory.

**Clarification:**

Confused why the returned value is an integer, but your answer is an array?

Note that the input array is passed in by **reference**, which means a modification to the input array will be known to the caller.

Internally you can think of this:

```
// nums is passed in by reference. (i.e., without making a copy)
int len = removeDuplicates(nums);

// any modification to nums in your function would be known by the caller.
// using the length returned by your function, it prints the first len elements.
for (int i = 0; i < len; i++) {
    print(nums[i]);
}
```

 

**Example 1:**

```
Input: nums = [1,1,1,2,2,3]
Output: 5, nums = [1,1,2,2,3]
Explanation: Your function should return length = 5, with the first five elements of nums being 1, 1, 2, 2 and 3 respectively. It doesn't matter what you leave beyond the returned length.
```

```python
def removeDuplicates(self, nums: List[int]) -> int:
    if len(nums) <= 2:
        return len(nums)

    slow = fast = 0
    prev = "#"
    count = 0

    while fast < len(nums):
        if nums[fast] != prev:
            count = 1
            prev = nums[fast]
            nums[slow] = nums[fast]
            slow += 1
            fast += 1
        elif count == 1:
            count += 1
            nums[slow] = nums[fast]
            slow += 1
            fast += 1
        else:
            fast += 1
    return slow
```

```python
class Solution(object):
    def removeDuplicates(self, nums):
        curr = count = 1
        while curr < len(nums):
            if nums[curr] == nums[curr - 1]:
                count += 1
            else:
                count = 1
            if count > 2:
                nums.pop(curr)
                curr -= 1
            curr += 1
        return len(nums)
```

```python
class Solution:
    def removeDuplicates(self, nums: List[int]) -> int:
        slow, count = 1, 1
        for fast in range(1, len(nums)):
            if nums[fast] == nums[fast - 1]:
                count += 1
            else:
                count = 1
            if count <= 2:
                nums[slow] = nums[fast]
                slow += 1
        return slow
```



#### 88. Merge Sorted Array

Given two sorted integer arrays `nums1` and `nums2`, merge `nums2` into `nums1` as one sorted array.

The number of elements initialized in `nums1` and `nums2` are `m` and `n` respectively. You may assume that `nums1` has a size equal to `m + n` such that it has enough space to hold additional elements from `nums2`.

**Example 1:**

```
Input: nums1 = [1,2,3,0,0,0], m = 3, nums2 = [2,5,6], n = 3
Output: [1,2,2,3,5,6]
```

```python
def merge(self, nums1: List[int], m: int, nums2: List[int], n: int) -> None:
    """
    Do not return anything, modify nums1 in-place instead.
    """
    while m > 0 and n > 0:
        if nums1[m-1] < nums2[n-1]:
            nums1[m+n-1] = nums2[n-1]
            n -= 1
        else:
            nums1[m-1], nums1[m+n-1] = nums1[m+n-1], nums1[m-1]
            m -= 1
    # if m >= n: all nums2 are moved to nums1 
    if n > m: # some head of nums2 are remaining and should be in head of nums1
        nums1[:n] = nums2[:n]
```



#### 167. Two Sum II - Input array is sorted

Given an array of integers `numbers` that is already ***sorted in non-decreasing order\***, find two numbers such that they add up to a specific `target` number.

Return *the indices of the two numbers (**1-indexed**) as an integer array* `answer` *of size* `2`*, where* `1 <= answer[0] < answer[1] <= numbers.length`.

The tests are generated such that there is **exactly one solution**. You **may not** use the same element twice.

 

**Example 1:**

```
Input: numbers = [2,7,11,15], target = 9
Output: [1,2]
Explanation: The sum of 2 and 7 is 9. Therefore index1 = 1, index2 = 2.
```

```python
class Solution:
    def twoSum(self, numbers: List[int], target: int) -> List[int]:
        start = 0
        end = len(numbers) - 1
        sum = 0
        
        while start < end:
            sum = numbers[start] + numbers[end]
            if sum > target:
                end = end - 1
            elif sum < target:
                start = start + 1
            else:
                return [start + 1, end + 1]
        
```



#### 209. Minimum Size Subarray Sum

Given an array of positive integers `nums` and a positive integer `target`, return the minimal length of a **contiguous subarray** `[numsl, numsl+1, ..., numsr-1, numsr]` of which the sum is greater than or equal to `target`. If there is no such subarray, return `0` instead.

 

**Example 1:**

```
Input: target = 7, nums = [2,3,1,2,4,3]
Output: 2
Explanation: The subarray [4,3] has the minimal length under the problem constraint.
```

```python
class Solution:
    def minSubArrayLen(self, target: int, nums: List[int]) -> int:
        length = float("inf")
        slow = 0
        sum = 0
        for fast in range(len(nums # if range(1, len(nums)): sum += nums[fast] doesnot include nums[0]
            sum += nums[fast]
            while sum >= target:
                length = min(length, fast - slow + 1)
                sum -= nums[slow]
                slow += 1
        return 0 if length == float("inf") else length
```



#### 283. Move Zeroes

Given an integer array `nums`, move all `0`'s to the end of it while maintaining the relative order of the non-zero elements.

**Note** that you must do this in-place without making a copy of the array.

 

**Example 1:**

```
Input: nums = [0,1,0,3,12]
Output: [1,3,12,0,0]
```

```python
class Solution:
    def moveZeroes(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        slow = 0
        fast = 0
        for fast in range(len(nums): #双指针把后面的移到前面一般fast都从slow开始，也就是说fast=slow开始
            if nums[fast] == 0:
                fast += 1
            else:
                nums[slow], nums[fast] = nums[fast], nums[slow]
                slow += 1
                fast += 1
```



#### 31. Next Permutation

Implement **next permutation**, which rearranges numbers into the lexicographically next greater permutation of numbers.

If such an arrangement is not possible, it must rearrange it as the lowest possible order (i.e., sorted in ascending order).

The replacement must be **[in place](http://en.wikipedia.org/wiki/In-place_algorithm)** and use only constant extra memory.

**Example 1:**

```
Input: nums = [1,2,3]
Output: [1,3,2]
```

**Example 2:**

```
Input: nums = [3,2,1]
Output: [1,2,3]
```

**Example 3:**

```
Input: nums = [1,1,5]
Output: [1,5,1]
```

```python
class Solution:
    def nextPermutation(self, nums: List[int]) -> None:
        i = len(nums) - 2
        while i >= 0 and nums[i] >= nums[i + 1]:
            i -= 1
        if i >= 0:
            j = len(nums) - 1
            while j >= 0 and nums[i] >= nums[j]:
                j -= 1
            nums[i], nums[j] = nums[j], nums[i]
        
        left, right = i + 1, len(nums) - 1
        while left < right:
            nums[left], nums[right] = nums[right], nums[left]
            left += 1
            right -= 1
```

#### 925. Long Pressed Name

```python
class Solution:
    def isLongPressedName(self, name: str, typed: str) -> bool:
        slow = 0
        fast = 0
        
        while fast < len(typed):
            if slow < len(name) and name[slow] == typed[fast]:
                slow += 1
                fast += 1
            elif fast > 0 and typed[fast] == typed[fast-1]:
                fast += 1
            else:
                return False
            
        return slow == len(name)
```





### Recursive

#### 112. Path Sum

Given the `root` of a binary tree and an integer `targetSum`, return `true` if the tree has a **root-to-leaf** path such that adding up all the values along the path equals `targetSum`.

A **leaf** is a node with no children.

 

**Example 1:**

![img](https://assets.leetcode.com/uploads/2021/01/18/pathsum1.jpg)

```
Input: root = [5,4,8,11,null,13,4,7,2,null,null,null,1], targetSum = 22
Output: true
```

```python
def hasPathSum(self, root: TreeNode, targetSum: int) -> bool:
    if root is None:
        return False
    if root.left is None and root.right is None:
        return root.val == targetSum
    return self.hasPathSum(root.left, targetSum - root.val) or self.hasPathSum(root.right, targetSum - root.val)
```







#### 22. Generate Parentheses

Given `n` pairs of parentheses, write a function to *generate all combinations of well-formed parentheses*.

 

**Example 1:**

```
Input: n = 3
Output: ["((()))","(()())","(())()","()(())","()()()"]
```

```python
class Solution(object):
    def generateParenthesis(self, n):
        def generate(l, r, p, result=[]):
            if l:
                generate(l-1, r, p+"(")
            if r > l:
                generate(l, r-1, p+")")
            if not r:
                result.append(p) 
            return result
        
        return generate(n, n, '')
```

```python
class Solution:
    def generateParenthesis(self, n: int) -> List[str]:
        if n == 0:
            return ['']
        ans = []
        for c in range(n):
            for left in self.generateParenthesis(c):
                for right in self.generateParenthesis(n-1-c):
                    ans.append('({}){}'.format(left, right))
        return ans

```





#### 50. Pow(x, n)

Implement [pow(x, n)](http://www.cplusplus.com/reference/valarray/pow/), which calculates `x` raised to the power `n` (i.e., `xn`).

 

**Example 1:**

```
Input: x = 2.00000, n = 10
Output: 1024.00000
```

```python
class Solution:
    def myPow(self, x: float, n: int) -> float:
        def quickMul(N):
            ans = 1.0
            # 贡献的初始值为 x
            x_contribute = x
            # 在对 N 进行二进制拆分的同时计算答案
            while N > 0:
                if N % 2 == 1:
                    # 如果 N 二进制表示的最低位为 1，那么需要计入贡献
                    ans *= x_contribute
                # 将贡献不断地平方
                x_contribute *= x_contribute
                # 舍弃 N 二进制表示的最低位，这样我们每次只要判断最低位即可
                N //= 2
            return ans
        
        return quickMul(n) if n >= 0 else 1.0 / quickMul(-n)

```







#### 172. Factorial Trailing Zeroes

Given an integer `n`, return *the number of trailing zeroes in `n!`*.

**Follow up:** Could you write a solution that works in logarithmic time complexity?

**Example 1:**

```
Input: n = 3
Output: 0
Explanation: 3! = 6, no trailing zero.
```

**Example 2:**

```
Input: n = 5
Output: 1
Explanation: 5! = 120, one trailing zero.
```

```python
class Solution:
    def trailingZeroes(self, n: int) -> int:
        ans = n // 5
        if ans == 0: return ans
        return ans + self.trailingZeroes(ans)
```







### Backtracking



#### 77*. Combinations

Given two integers `n` and `k`, return *all possible combinations of* `k` *numbers out of the range* `[1, n]`.

You may return the answer in **any order**.

 

**Example 1:**

```
Input: n = 4, k = 2
Output:
[
  [2,4],
  [3,4],
  [2,3],
  [1,2],
  [1,3],
  [1,4],
]
```

```python
class Solution:
    def combine(self, n: int, k: int) -> List[List[int]]:
        # return list(itertools.combinations(range(1,n+1),k))
        res=[]  #存放符合条件结果的集合
        path=[]  #用来存放符合条件结果
        def backtrack(n,k,startIndex):
            if len(path) == k:
                res.append(path[:])
                return 
            for i in range(startIndex, n-(k-len(path))+2):  #优化的地方
                path.append(i)  #处理节点 
                backtrack(n, k, i+1)  #递归
                path.pop()  #回溯，撤销处理的节点
        backtrack(n,k,1)
        return res

```

#### 216. Combination Sum III

Find all valid combinations of `k` numbers that sum up to `n` such that the following conditions are true:

- Only numbers `1` through `9` are used.
- Each number is used **at most once**.

Return *a list of all possible valid combinations*. The list must not contain the same combination twice, and the combinations may be returned in any order.

 

**Example 1:**

```
Input: k = 3, n = 7
Output: [[1,2,4]]
Explanation:
1 + 2 + 4 = 7
There are no other valid combinations.
```

```python
class Solution:
    def combinationSum3(self, k: int, n: int) -> List[List[int]]:
        res = []  #存放结果集
        self.findallPath(n,k,0,1,[],res)
        return res
    
    def findallPath(self,n,k,sum,startIndex,path,res):
            if sum > n: return  #剪枝操作
            if sum == n and len(path) == k:  #如果path.size() == k 但sum != n 直接返回
                return res.append(path[:])
            for i in range(startIndex,9-(k-len(path))+2):  #剪枝操作
                path.append(i)
                sum += i 
                self.findallPath(n,k,sum,i+1,path,res)  #注意i+1调整startIndex
                sum -= i  #回溯
                path.pop()  #回溯
```

```python
import itertools
class Solution:
    def combinationSum3(self, k: int, n: int) -> List[List[int]]:
        allNumbers=[i for i in range(1,10)]
        res=[]
        for item in itertools.combinations(allNumbers,k):
            if sum(item)==n:
                res.append(item)
                
        return res  
```



#### 17. Letter Combinations of a Phone Number

Given a string containing digits from `2-9` inclusive, return all possible letter combinations that the number could represent. Return the answer in **any order**.

A mapping of digit to letters (just like on the telephone buttons) is given below. Note that 1 does not map to any letters.

![img](https://upload.wikimedia.org/wikipedia/commons/thumb/7/73/Telephone-keypad2.svg/200px-Telephone-keypad2.svg.png)

 

**Example 1:**

```
Input: digits = "23"
Output: ["ad","ae","af","bd","be","bf","cd","ce","cf"]
```

```python
# Algorithm: Greedy
def letterCombinations(self, digits: str) -> List[str]:
    if len(digits) == 0:
        return []
    digits_map = {
        0: "0",
        1: "1",
        2: "abc",
        3: "def",
        4: "ghi",
        5: "jkl",
        6: "mno",
        7: "pqrs",
        8: "tuv",
        9: "wxyz"
    }
    result = [""]

    for digit in digits:
        temp_list = []
        for ch in digits_map[int(digit)]:
            for st in result:
                temp_list.append(st + ch)
        result = temp_list
    return result
```

```python
# Algorithm: DFS
class Solution:
    def letterCombinations(self, digits):
        if len(digits) == 0:
            return []
        dic = {"2":"abc", '3':"def", '4':"ghi", '5':"jkl", '6':"mno", '7':"pqrs", '8':"tuv", '9':"wxyz"}
        result = []
        self.dfs(digits, dic, 0, "", result)
        return result

    def dfs(self, digits, dic, index, path, result):
        if len(path) == len(digits):
            result.append(path)
            return

        for j in dic[digits[index]]:
            path = path + j
            self.dfs(digits, dic, index+1, path, result)
            path = path[:-1]
```

#### 39. Combination Sum

Given an array of **distinct** integers `candidates` and a target integer `target`, return *a list of all **unique combinations** of* `candidates` *where the chosen numbers sum to* `target`*.* You may return the combinations in **any order**.

The **same** number may be chosen from `candidates` an **unlimited number of times**. Two combinations are unique if the frequency of at least one of the chosen numbers is different.

It is **guaranteed** that the number of unique combinations that sum up to `target` is less than `150` combinations for the given input.

**Example 1:**

```
Input: candidates = [2,3,6,7], target = 7
Output: [[2,2,3],[7]]
Explanation:
2 and 3 are candidates, and 2 + 2 + 3 = 7. Note that 2 can be used multiple times.
7 is a candidate, and 7 = 7.
These are the only two combinations.
```

```python
def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
    result = []
    candidates.sort()
    self.dfs(candidates, target, 0, [], result)

    return result

def dfs(self, nums, target, index, path, result):
    if target < 0:
        return

    if target == 0:
        result.append(path)
        return

    for i in range(index, len(nums)):
        self.dfs(nums, target - nums[i], i, path + [nums[i]], result)
```

#### 40. Combination Sum II

Given a collection of candidate numbers (`candidates`) and a target number (`target`), find all unique combinations in `candidates` where the candidate numbers sum to `target`.

Each number in `candidates` may only be used **once** in the combination.

**Note:** The solution set must not contain duplicate combinations.

**Example 1:**

```
Input: candidates = [10,1,2,7,6,1,5], target = 8
Output: 
[
[1,1,6],
[1,2,5],
[1,7],
[2,6]
]
```

```python
def combinationSum2(self, candidates: List[int], target: int) -> List[List[int]]:
    result = []
    candidates.sort()

    self.dfs(candidates, target, 0, [], result)

    return result

def dfs(self, nums, target, index, path, result):
    if target == 0:
        result.append(path)
        return 

    for i in range(index, len(nums)):
        if i > index and nums[i] == nums[i-1]:
            continue
        if nums[i] > target:
            break
        self.dfs(nums, target - nums[i], i + 1, path + [nums[i]], result)
```

```python
class Solution:
    def combinationSum2(self, candidates: List[int], target: int) -> List[List[int]]:
        path = []
        res = []
        
        candidates.sort()
        
        def dfs(candidates, target, index):
            if sum(path) > target: return
            if sum(path) == target:
                res.append(path[:])
                return
            for i in range(index, len(candidates)):
                if i > index and candidates[i] == candidates[i - 1]:
                    continue
                path.append(candidates[i])
                dfs(candidates, target, i + 1)
                path.pop()
                
        dfs(candidates, target, 0)
        
        return res
```



#### 131. Palindrome Partitioning

Given a string `s`, partition `s` such that every substring of the partition is a **palindrome**. Return all possible palindrome partitioning of `s`.

A **palindrome** string is a string that reads the same backward as forward.

**Example 1:**

```
Input: s = "aab"
Output: [["a","a","b"],["aa","b"]]
```

```python
class Solution:
    def partition(self, s: str) -> List[List[str]]:
        def dfs(s, index):
            if index == len(s):
                return res.append(path[:])

            for i in range(index, len(s)):

                if s[index:i+1] == s[index:i+1][::-1]:
                    path.append(s[index:i+1])
                    dfs(s, i+1)
                    path.pop()
                
        res = []
        path = []
        dfs(s, 0)
        
        return res
```

```python
class Solution:
    def partition(self, s: str) -> List[List[str]]:
        res = []  
        path = []  #放已经回文的子串
        def backtrack(s,startIndex):
            if startIndex >= len(s):  #如果起始位置已经大于s的大小，说明已经找到了一组分割方案了
                return res.append(path[:])
            for i in range(startIndex,len(s)):
                p = s[startIndex:i+1]  #获取[startIndex,i+1]在s中的子串
                if p == p[::-1]: path.append(p)  #是回文子串
                else: continue  #不是回文，跳过
                backtrack(s,i+1)  #寻找i+1为起始位置的子串
                path.pop()  #回溯过程，弹出本次已经填在path的子串
        backtrack(s,0)
        return res
```



#### 93. Restore IP Addresses

Given a string `s` containing only digits, return all possible valid IP addresses that can be obtained from `s`. You can return them in **any** order.

A **valid IP address** consists of exactly four integers, each integer is between `0` and `255`, separated by single dots and cannot have leading zeros. For example, "0.1.2.201" and "192.168.1.1" are **valid** IP addresses and "0.011.255.245", "192.168.1.312" and "192.168@1.1" are **invalid** IP addresses. 

 

**Example 1:**

```
Input: s = "25525511135"
Output: ["255.255.11.135","255.255.111.35"]
```

```python
class Solution:
    def restoreIpAddresses(self, s: str) -> List[str]:
        res = []
        n = len(s)
        def dfs(s, dot, pos, path):
            if dot >= 4: 
                if pos == n:
                    res.append(path[1:])
                return
            if pos >= n: # 剪枝
                return
            for i in range(1, min(3, n-pos) + 1): # 剪枝
                temp = s[pos:pos+i]
                if int(temp) <= 255:
                    if temp[0] == "0" and len(temp) > 1:
                        continue
                    dfs(s, dot+1, pos+ len(temp), path + "." + temp)
        
        dfs(s,0,0,"")
        return res
```

```python
class Solution(object):
    def restoreIpAddresses(self, s):
        """
        :type s: str
        :rtype: List[str]
        """
        ans = []
        path = []
        def backtrack(path, startIndex):
            if len(path) == 4:
                if startIndex == len(s):
                    ans.append(".".join(path[:]))
                    return
            for i in range(startIndex+1, min(startIndex+4, len(s)+1)):  # 剪枝
                string = s[startIndex:i]
                if not 0 <= int(string) <= 255:
                    continue
                if not string == "0" and not string.lstrip('0') == string:
                    continue
                path.append(string)
                backtrack(path, i)
                path.pop()

        backtrack([], 0)
        return ans
```



#### 78*. Subsets

Given an integer array `nums` of **unique** elements, return *all possible subsets (the power set)*.

The solution set **must not** contain duplicate subsets. Return the solution in **any order**.

 

**Example 1:**

```
Input: nums = [1,2,3]
Output: [[],[1],[2],[1,2],[3],[1,3],[2,3],[1,2,3]]
```

```python
class Solution:
    def subsets(self, nums: List[int]) -> List[List[int]]:
        res = []  
        path = []  
        def backtrack(nums,startIndex):
            res.append(path[:])  #收集子集，要放在终止添加的上面，否则会漏掉自己
            for i in range(startIndex,len(nums)):  
                #当startIndex已经大于数组的长度了，就终止了，for循环本来也结束了，所以不需要终止条件
                path.append(nums[i])
                backtrack(nums,i+1)  #递归
                path.pop()  #回溯
        backtrack(nums,0)
        return res
```



#### 90. Subsets II

Given an integer array `nums` that may contain duplicates, return *all possible subsets (the power set)*.

The solution set **must not** contain duplicate subsets. Return the solution in **any order**.

 

**Example 1:**

```
Input: nums = [1,2,2]
Output: [[],[1],[1,2],[1,2,2],[2],[2,2]]
```

```python
class Solution:
    def subsetsWithDup(self, nums: List[int]) -> List[List[int]]:
        res = []
        n = len(nums)
        nums.sort()
        
        def dfs(nums, index, path):
            res.append(path[:])
            
            for i in range(index, n):
                if i > index and nums[i] == nums[i-1]:
                    continue
                dfs(nums, i+1, path + [nums[i]])
                
        dfs(nums, 0, [])
        return res
```





#### 491. Increasing Subsequences

Given an integer array `nums`, return all the different possible increasing subsequences of the given array with **at least two elements**. You may return the answer in **any order**.

The given array may contain duplicates, and two equal integers should also be considered a special case of increasing sequence.

 

**Example 1:**

```
Input: nums = [4,6,7,7]
Output: [[4,6],[4,6,7],[4,6,7,7],[4,7],[4,7,7],[6,7],[6,7,7],[7,7]]
```

```python
class Solution:
    def findSubsequences(self, nums: List[int]) -> List[List[int]]:
        res = []
        n = len(nums)
        def dfs(nums, index, path):
            if len(path) >= 2:
                res.append(path[:])
                
            used = []
            for i in range(index, n):
                if nums[i] in used:
                    continue
                if not path or nums[i] >= path[-1]:
                    used.append(nums[i])
                    dfs(nums, i+1, path + [nums[i]])
                    
        dfs(nums, 0, [])
        
        return res
```

#### 46. Permutations

Given an array `nums` of distinct integers, return *all the possible permutations*. You can return the answer in **any order**.

**Example 1:**

```
Input: nums = [1,2,3]
Output: [[1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2],[3,2,1]]
```

```python
def permute(self, nums: List[int]) -> List[List[int]]:
    if len(nums) == 1:
        return [nums]
    output = []
    for i, num in enumerate(nums):
        n = nums[:i] + nums[i+1:]

        for y in self.permute(n):
            output.append([num] + y)

    return output
```

```python 
class Solution:
    def permute(self, nums: List[int]) -> List[List[int]]:
        res = []
        
        def dfs(nums, index, path):
            if not nums:
                res.append(path[:])
                
            for i in range(0, len(nums)):
                dfs(nums[:i] + nums[i+1:], index+1, path + [nums[i]])
        
        dfs(nums,0,[])
        return res
```

```python
class Solution:
    def permute(self, nums: List[int]) -> List[List[int]]:
        res = []  #存放符合条件结果的集合
        path = []  #用来存放符合条件的结果
        def backtrack(nums):
            if len(path) == len(nums):
                return res.append(path[:])  #此时说明找到了一组
            for i in range(0,len(nums)):
                if nums[i] in path:  #path里已经收录的元素，直接跳过
                    continue
                path.append(nums[i])
                backtrack(nums)  #递归
                path.pop()  #回溯
        backtrack(nums)
        return res
```

```python
class Solution:
    def permute(self, nums: List[int]) -> List[List[int]]:
        res = []  #存放符合条件结果的集合
        path = []  #用来存放符合条件的结果
        used = []  #用来存放已经用过的数字
        def backtrack(nums,used):
            if len(path) == len(nums):
                return res.append(path[:])  #此时说明找到了一组
            for i in range(0,len(nums)):
                if nums[i] in used:
                    continue  #used里已经收录的元素，直接跳过
                path.append(nums[i])
                used.append(nums[i])
                backtrack(nums,used)
                used.pop()
                path.pop()
        backtrack(nums,used)
        return res
```





#### 47*. Permutations II

Given a collection of numbers, `nums`, that might contain duplicates, return *all possible unique permutations **in any order**.*

**Example 1:**

```
Input: nums = [1,1,2]
Output:
[[1,1,2],
 [1,2,1],
 [2,1,1]]
```

**Example 2:**

```
Input: nums = [1,2,3]
Output: [[1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2],[3,2,1]]
```

```python
class Solution:
    def permuteUnique(self, nums: List[int]) -> List[List[int]]:
        def dfs(nums, path):
            if n == len(path):
                res.append(path)
                return

            for i in range(n):
            # used[i - 1] == true，说明同一树支nums[i - 1]使用过
            # used[i - 1] == false，说明同一树层nums[i - 1]使用过
            # 如果同一树层nums[i - 1]使用过则直接跳过
            # 树层去重：重复数字只会在同一树枝出现，不会在同一树层出现
            # 由于i是从小到大遍历，重复数字永远用在上一层刚刚出现的同一数字的紧接着下一个（看例子（1,1,1））
                if not used[i]:
                    if (i > 0 and nums[i] == nums[i-1] and not used[i-1]):
                        continue
                    used[i] = True
                    dfs(nums, path + [nums[i]])
                    used[i] = False
        res = []
        n = len(nums)
        used = [False] * len(nums)
        nums.sort()
        dfs(nums, [])

        return res

```







#### 332. Reconstruct Itinerary

You are given a list of airline `tickets` where `tickets[i] = [fromi, toi]` represent the departure and the arrival airports of one flight. Reconstruct the itinerary in order and return it.

All of the tickets belong to a man who departs from `"JFK"`, thus, the itinerary must begin with `"JFK"`. If there are multiple valid itineraries, you should return the itinerary that has the smallest lexical order when read as a single string.

- For example, the itinerary `["JFK", "LGA"]` has a smaller lexical order than `["JFK", "LGB"]`.

You may assume all tickets form at least one valid itinerary. You must use all the tickets once and only once.

 

**Example 1:**

![img](https://assets.leetcode.com/uploads/2021/03/14/itinerary1-graph.jpg)

```
Input: tickets = [["MUC","LHR"],["JFK","MUC"],["SFO","SJC"],["LHR","SFO"]]
Output: ["JFK","MUC","LHR","SFO","SJC"]
```

```python
class Solution:
    def findItinerary(self, tickets: List[List[str]]) -> List[str]:
        # defaultdic(list) 是为了方便直接append
        tickets_dict = defaultdict(list)
        for item in tickets:
            tickets_dict[item[0]].append(item[1])
        '''
        tickets_dict里面的内容是这样的
         {'JFK': ['SFO', 'ATL'], 'SFO': ['ATL'], 'ATL': ['JFK', 'SFO']})
        '''
        path = ["JFK"]
        def backtracking(start_point):
            # 终止条件
            if len(path) == len(tickets) + 1:
                return True
            tickets_dict[start_point].sort()
            for _ in tickets_dict[start_point]:
                # 必须及时删除，避免出现死循环
                end_point = tickets_dict[start_point].pop(0)
                path.append(end_point)
                # 只要找到一个就可以返回了
                if backtracking(end_point):
                    return True
                path.pop()
                tickets_dict[start_point].append(end_point)

        backtracking("JFK")
        return path
```

#### 51. N-Queens

The **n-queens** puzzle is the problem of placing `n` queens on an `n x n` chessboard such that no two queens attack each other.

Given an integer `n`, return *all distinct solutions to the **n-queens puzzle***. You may return the answer in **any order**.

Each solution contains a distinct board configuration of the n-queens' placement, where `'Q'` and `'.'` both indicate a queen and an empty space, respectively.

 

**Example 1:**

![img](https://assets.leetcode.com/uploads/2020/11/13/queens.jpg)

```
Input: n = 4
Output: [[".Q..","...Q","Q...","..Q."],["..Q.","Q...","...Q",".Q.."]]
Explanation: There exist two distinct solutions to the 4-queens puzzle as shown above
```

```python
class Solution:
    def solveNQueens(self, n: int) -> List[List[str]]:
        if not n: return []
        board = [['.'] * n for _ in range(n)]
        res = []
        def isVaild(board,row, col):
            #判断同一列是否冲突
            for i in range(len(board)):
                if board[i][col] == 'Q':
                    return False
            # 判断左上角是否冲突
            i = row -1
            j = col -1
            while i>=0 and j>=0:
                if board[i][j] == 'Q':
                    return False
                i -= 1
                j -= 1
            # 判断右上角是否冲突
            i = row - 1
            j = col + 1
            while i>=0 and j < len(board):
                if board[i][j] == 'Q':
                    return False
                i -= 1
                j += 1
            return True

        def backtracking(board, row, n):
            # 如果走到最后一行，说明已经找到一个解
            if row == n:
                temp_res = []
                for temp in board:
                    temp_str = "".join(temp)
                    temp_res.append(temp_str)
                res.append(temp_res)
            for col in range(n):
                if not isVaild(board, row, col):
                    continue
                board[row][col] = 'Q'
                backtracking(board, row+1, n)
                board[row][col] = '.'
        backtracking(board, 0, n)
        return res
```



#### 37. Sudoku Solver

Write a program to solve a Sudoku puzzle by filling the empty cells.

A sudoku solution must satisfy **all of the following rules**:

1. Each of the digits `1-9` must occur exactly once in each row.
2. Each of the digits `1-9` must occur exactly once in each column.
3. Each of the digits `1-9` must occur exactly once in each of the 9 `3x3` sub-boxes of the grid.

The `'.'` character indicates empty cells.

 

**Example 1:**

![img](https://upload.wikimedia.org/wikipedia/commons/thumb/f/ff/Sudoku-by-L2G-20050714.svg/250px-Sudoku-by-L2G-20050714.svg.png)

```
Input: board = [["5","3",".",".","7",".",".",".","."],["6",".",".","1","9","5",".",".","."],[".","9","8",".",".",".",".","6","."],["8",".",".",".","6",".",".",".","3"],["4",".",".","8",".","3",".",".","1"],["7",".",".",".","2",".",".",".","6"],[".","6",".",".",".",".","2","8","."],[".",".",".","4","1","9",".",".","5"],[".",".",".",".","8",".",".","7","9"]]
Output: [["5","3","4","6","7","8","9","1","2"],["6","7","2","1","9","5","3","4","8"],["1","9","8","3","4","2","5","6","7"],["8","5","9","7","6","1","4","2","3"],["4","2","6","8","5","3","7","9","1"],["7","1","3","9","2","4","8","5","6"],["9","6","1","5","3","7","2","8","4"],["2","8","7","4","1","9","6","3","5"],["3","4","5","2","8","6","1","7","9"]]
Explanation: The input board is shown above and the only valid solution is shown below:
```

```python
class Solution:
    def solveSudoku(self, board: List[List[str]]) -> None:
        """
        Do not return anything, modify board in-place instead.
        """
        def backtrack(board):
            for i in range(len(board)):  #遍历行
                for j in range(len(board[0])):  #遍历列
                    if board[i][j] != ".": continue
                    for k in range(1,10):  #(i, j) 这个位置放k是否合适
                        if isValid(i,j,k,board):
                            board[i][j] = str(k)  #放置k
                            if backtrack(board): return True  #如果找到合适一组立刻返回
                            board[i][j] = "."  #回溯，撤销k
                    return False  #9个数都试完了，都不行，那么就返回false
            return True  #遍历完没有返回false，说明找到了合适棋盘位置了
        
        def isValid(row,col,val,board):
            for i in range(9):  #判断行里是否重复
                if board[row][i] == str(val):
                    return False
            for j in range(9):  #判断列里是否重复
                if board[j][col] == str(val):
                    return False
            startRow = (row // 3) * 3
            startcol = (col // 3) * 3
            for i in range(startRow,startRow + 3):  #判断9方格里是否重复
                for j in range(startcol,startcol + 3):
                    if board[i][j] == str(val):
                        return False
            return True
        backtrack(board)
```



#### 79. Word Search



```python
def exist(self, board: List[List[str]], word: str) -> bool:
    for i in range(len(board)):
        for j in range(len(board[0])):
            if self.helper(board, word, 0, i, j):
                return True

    return False

def helper(self, board, word, wordIndex, i, j):
    if wordIndex == len(word):
        return True
    if i < 0 or i >= len(board) or j < 0 or j >= len(board[0]) or word[wordIndex] != board[i][j]:
        return False
    board[i][j] = "#"

    found =    self.helper(board, word, wordIndex+1, i+1, j) \
            or self.helper(board, word, wordIndex+1, i, j+1) \
            or self.helper(board, word, wordIndex+1, i-1, j) \
            or self.helper(board, word, wordIndex+1, i, j-1) 

    board[i][j] = word[wordIndex]

    return found
```















### Greedy

#### 45. Jump Game II

Given an array of non-negative integers `nums`, you are initially positioned at the first index of the array.

Each element in the array represents your maximum jump length at that position.

Your goal is to reach the last index in the minimum number of jumps.

You can assume that you can always reach the last index.

 

**Example 1:**

```
Input: nums = [2,3,1,1,4]
Output: 2
Explanation: The minimum number of jumps to reach the last index is 2. Jump 1 step from index 0 to 1, then 3 steps to the last index.
```

```python
class Solution:
    def jump(self, nums: List[int]) -> int:
        n = len(nums)
        maxPos = end = step = 0 # end表示候选集边界，是在候选集里选最远可达位置
        # 候选集可以理解为，我这一步可以跳到的位置集合，但具体跳到哪一个，取决于它们各自的最远可达maxPos
        for i in range(n-1):
            if maxPos >= i:  # 所在位置未出最远可达 
                maxPos = max(maxPos, i + nums[i])  # 更新最远可达
                if i >= end:  # 所在位置超出候选集边界，这时候必须跳动，跳动到拥有最远maxPos的位置
                    end = maxPos # 更新候选集边界
                    step += 1
        return step
```



#### 53. Maximum Subarray

Given an integer array `nums`, find the contiguous subarray (containing at least one number) which has the largest sum and return *its sum*.

**Example 1:**

```
Input: nums = [-2,1,-3,4,-1,2,1,-5,4]
Output: 6
Explanation: [4,-1,2,1] has the largest sum = 6.
```

```python
def maxSubArray(self, nums: List[int]) -> int:
    for i in range(0, len(nums)-1):
        nums[i+1] += max(0, nums[i])
    return max(nums)
```

```python
def maxSubArray(self, nums: List[int]) -> int:
    if max(nums) < 0:
        return max(nums)

    local_max, global_max = 0, 0
    for num in nums:
        local_max = max(0, local_max + num)
        global_max = max(global_max, local_max)
            
    return global_max
```

```python
class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        if len(nums) == 0:
            return 0
        dp = [0] * len(nums)
        dp[0] = nums[0]
        result = dp[0]
        for i in range(1, len(nums)):
            dp[i] = max(dp[i-1] + nums[i], nums[i]) #状态转移公式
            result = max(result, dp[i]) #result 保存dp[i]的最大值
        return result
```



#### 55. Jump Game

Given an array of non-negative integers `nums`, you are initially positioned at the **first index** of the array.

Each element in the array represents your maximum jump length at that position.

Determine if you are able to reach the last index.

 

**Example 1:**

```
Input: nums = [2,3,1,1,4]
Output: true
Explanation: Jump 1 step from index 0 to 1, then 3 steps to the last index.
```

```python
def canJump(self, nums: List[int]) -> bool:
    reach = 0
    for i in range(len(nums)):
        if i > reach:
            return False
        if nums[i] + i > reach: 
            reach = nums[i] + i
    return True
```



#### 134. Gas Station

There are `n` gas stations along a circular route, where the amount of gas at the `ith` station is `gas[i]`.

You have a car with an unlimited gas tank and it costs `cost[i]` of gas to travel from the `ith` station to its next `(i + 1)th` station. You begin the journey with an empty tank at one of the gas stations.

Given two integer arrays `gas` and `cost`, return *the starting gas station's index if you can travel around the circuit once in the clockwise direction, otherwise return* `-1`. If there exists a solution, it is **guaranteed** to be **unique**

 

**Example 1:**

```
Input: gas = [1,2,3,4,5], cost = [3,4,5,1,2]
Output: 3
Explanation:
Start at station 3 (index 3) and fill up with 4 unit of gas. Your tank = 0 + 4 = 4
Travel to station 4. Your tank = 4 - 1 + 5 = 8
Travel to station 0. Your tank = 8 - 2 + 1 = 7
Travel to station 1. Your tank = 7 - 3 + 2 = 6
Travel to station 2. Your tank = 6 - 4 + 3 = 5
Travel to station 3. The cost is 5. Your gas is just enough to travel back to station 3.
Therefore, return 3 as the starting index.
```

```python
class Solution:
    def canCompleteCircuit(self, gas: List[int], cost: List[int]) -> int:
        start = 0
        curSum = 0
        totalSum = 0
        for i in range(len(gas)):
            curSum += gas[i] - cost[i]
            totalSum += gas[i] - cost[i]
            if curSum < 0:
                curSum = 0
                start = i + 1
        if totalSum < 0: return -1
        return start
```



#### 135. Candy

There are `n` children standing in a line. Each child is assigned a rating value given in the integer array `ratings`.

You are giving candies to these children subjected to the following requirements:

- Each child must have at least one candy.
- Children with a higher rating get more candies than their neighbors.

Return *the minimum number of candies you need to have to distribute the candies to the children*.

 

**Example 1:**

```
Input: ratings = [1,0,2]
Output: 5
Explanation: You can allocate to the first, second and third child with 2, 1, 2 candies respectively.
```

```python
class Solution:
    def candy(self, ratings: List[int]) -> int:
        candyVec = [1] * len(ratings)
        for i in range(1, len(ratings)):
            if ratings[i] > ratings[i - 1]:
                candyVec[i] = candyVec[i - 1] + 1
        for j in range(len(ratings) - 2, -1, -1):
            if ratings[j] > ratings[j + 1]:
                candyVec[j] = max(candyVec[j], candyVec[j + 1] + 1)
        return sum(candyVec)
```



### DP

#### 5. Longest Palindromic Substring

Given a string `s`, return *the longest palindromic substring* in `s`.

**Example 1:**

```
Input: s = "babad"
Output: "bab"
Note: "aba" is also a valid answer.
```

**Example 2:**

```
Input: s = "cbbd"
Output: "bb"
```

**Example 3:**

```
Input: s = "a"
Output: "a"
```

**Example 4:**

```
Input: s = "ac"
Output: "a"
```

```python
class Solution:
    def longestPalindrome(self, s: str) -> str:
        Palindrome = ""
        for i in range(len(s)):
            temp1 = self.getlongestPalindrome(s, i, i)
            if len(temp1) > len(Palindrome):
                Palindrome = temp1
            temp2 = self.getlongestPalindrome(s, i, i+1)
            if len(temp2) > len(Palindrome):
                Palindrome = temp2
        return Palindrome
    
    def getlongestPalindrome(self, s, l, r):
        while l >= 0 and r < len(s) and s[l] == s[r]:
            l -= 1
            r += 1
        return s[l+1 : r]
```



#### 62. Unique Paths

A robot is located at the top-left corner of a `m x n` grid (marked 'Start' in the diagram below).

The robot can only move either down or right at any point in time. The robot is trying to reach the bottom-right corner of the grid (marked 'Finish' in the diagram below).

How many possible unique paths are there?

 

**Example 1:**

![img](https://assets.leetcode.com/uploads/2018/10/22/robot_maze.png)

```
Input: m = 3, n = 7
Output: 28
```



```python
# math
class Solution:
    def uniquePaths(self, m: int, n: int) -> int:
        return comb(m+n-2, m-1)
```

```python
# DP
class Solution:
    def uniquePaths(self, m: int, n: int) -> int:
        f = [[1] * n] + [[1] + [0] * (n - 1) for _ in range(m - 1)]
        for i in range(1, m):
            for j in range(1, n):
                f[i][j] = f[i - 1][j] + f[i][j - 1]
        return f[m - 1][n - 1]
```





#### 70. Climbing Stairs

You are climbing a staircase. It takes `n` steps to reach the top.

Each time you can either climb `1` or `2` steps. In how many distinct ways can you climb to the top?

**Example 1:**

```
Input: n = 2
Output: 2
Explanation: There are two ways to climb to the top.
1. 1 step + 1 step
2. 2 steps
```



```python
def climbStairs(self, n):
    a,b = 1,0
    for _ in range(n):
        a,b = a+b,a
    return a
```

```python
def climbStairs(self, n: int) -> int:
    dp = [0]*(n+1)
    dp[0] = 1
    dp[1] = 2
    for i in range(2,n):
        dp[i] = dp[i-1] + dp[i-2]
    return dp[n-1]
```

```python
# Each time you can either climb `1` ... `m` steps.
class Solution:
    def climbStairs(self, n: int) -> int:
        dp = [0]*(n + 1)
        dp[0] = 1
        m = 2
        # 排列问题
        for j in range(n + 1): # 遍历背包
            for step in range(1, m + 1): # 遍历物品
                if j >= step:
                    dp[j] += dp[j - step]
        return dp[n]
```



#### 63. Unique Paths II

A robot is located at the top-left corner of a `m x n` grid (marked 'Start' in the diagram below).

The robot can only move either down or right at any point in time. The robot is trying to reach the bottom-right corner of the grid (marked 'Finish' in the diagram below).

Now consider if some obstacles are added to the grids. How many unique paths would there be?

An obstacle and space is marked as `1` and `0` respectively in the grid.

 

**Example 1:**

![img](https://assets.leetcode.com/uploads/2020/11/04/robot1.jpg)

```
Input: obstacleGrid = [[0,0,0],[0,1,0],[0,0,0]]
Output: 2
Explanation: There is one obstacle in the middle of the 3x3 grid above.
There are two ways to reach the bottom-right corner:
1. Right -> Right -> Down -> Down
2. Down -> Down -> Right -> Right
```

```python
class Solution:
    def uniquePathsWithObstacles(self, obstacleGrid: List[List[int]]) -> int:
        m = len(obstacleGrid)
        n = len(obstacleGrid[0])

        if obstacleGrid[0][0] == 1 or obstacleGrid[m-1][n-1] == 1:
            return 0
        
        obstacleGrid[0][0] = 1
        
        for j in range(1, n):
            obstacleGrid[0][j] = int(obstacleGrid[0][j] == 0 and obstacleGrid[0][j-1] == 1)

        for i in range(1, m):
            obstacleGrid[i][0] = int(obstacleGrid[i][0] == 0 and obstacleGrid[i-1][0] == 1)
            
        for i in range(1, m):
            for j in range(1, n):
                if obstacleGrid[i][j] == 0:
                    obstacleGrid[i][j] =  obstacleGrid[i-1][j] +  obstacleGrid[i][j-1]
                else:
                    obstacleGrid[i][j] = 0
                    
        return obstacleGrid[m-1][n-1] 
```



#### 64. Minimum Path Sum

Given a `m x n` `grid` filled with non-negative numbers, find a path from top left to bottom right, which minimizes the sum of all numbers along its path.

**Note:** You can only move either down or right at any point in time.

 

**Example 1:**

![img](https://assets.leetcode.com/uploads/2020/11/05/minpath.jpg)

```
Input: grid = [[1,3,1],[1,5,1],[4,2,1]]
Output: 7
Explanation: Because the path 1 → 3 → 1 → 1 → 1 minimizes the sum.
```

```python
class Solution:
    def minPathSum(self, grid: List[List[int]]) -> int:
        m = len(grid)
        n = len(grid[0])
        
        for j in range(1, n):
            grid[0][j] = grid[0][j] + grid[0][j-1]
        
        for i in range(1, m):
            grid[i][0] = grid[i][0] + grid[i-1][0]
            
        for i in range(1, m):
            for j in range(1, n):
                grid[i][j] = grid[i][j] + min(grid[i-1][j], grid[i][j-1])
                
        return grid[m-1][n-1]
```

```python
class Solution:
    def minPathSum(self, grid):
        dp = [float('inf')] * (len(grid[0]) + 1)
        dp[1] = 0
        for row in grid:
            for idx, num in enumerate(row):
                dp[idx + 1] = min(dp[idx], dp[idx + 1]) + num
        return dp[-1]
```



#### 20. Valid Parentheses

Given a string `s` containing just the characters `'('`, `')'`, `'{'`, `'}'`, `'['` and `']'`, determine if the input string is valid.

An input string is valid if:

1. Open brackets must be closed by the same type of brackets.
2. Open brackets must be closed in the correct order.

 

**Example 1:**

```
Input: s = "()"
Output: true
```

```python
class Solution:
    def isValid(self, s: str) -> bool:
        stack = []  # 保存还未匹配的左括号
        mapping = {")": "(", "]": "[", "}": "{"}
        for i in s:
            if i in "([{":  # 当前是左括号，则入栈
                stack.append(i)
            elif stack and stack[-1] == mapping[i]:  # 当前是配对的右括号则出栈
                stack.pop()
            else:  # 不是匹配的右括号或者没有左括号与之匹配，则返回false
                return False
        return stack == []  # 最后必须正好把左括号匹配完
```



#### 96. Unique Binary Search Trees

Given an integer `n`, return *the number of structurally unique **BST'**s (binary search trees) which has exactly* `n` *nodes of unique values from* `1` *to* `n`.

 

**Example 1:**

![img](https://assets.leetcode.com/uploads/2021/01/18/uniquebstn3.jpg)

```
Input: n = 3
Output: 5
```

**Example 2:**

```
Input: n = 1
Output: 1
```

```python
class Solution:
    def numTrees(self, n: int) -> int:
        dp = [0] * (n + 1)
        dp[0], dp[1] = 1, 1
        for i in range(2, n+1):
            for j in range(i):
                dp[i] += dp[j]*dp[i-j-1]
        return dp[n]
```

#### 152. Maximum Product Subarray

Given an integer array `nums`, find a contiguous non-empty subarray within the array that has the largest product, and return *the product*.

It is **guaranteed** that the answer will fit in a **32-bit** integer.

A **subarray** is a contiguous subsequence of the array.

 

**Example 1:**

```
Input: nums = [2,3,-2,4]
Output: 6
Explanation: [2,3] has the largest product 6.
```

```python
class Solution:
    def maxProduct(self, nums: List[int]) -> int:
        n = len(nums)
        maxdp = [0] * n
        mindp = [0] * n
        maxdp[0] = nums[0]
        mindp[0] = nums[0]
        for i in range(1, n):
            temp = [nums[i], maxdp[i-1]*nums[i], mindp[i-1]*nums[i]]
            mindp[i] = min(temp)
            maxdp[i] = max(temp)
        return max(maxdp)
```

```python
class Solution:
    def maxProduct(self, nums: List[int]) -> int:
        n = len(nums)
        maxdp, mindp, ans = nums[0], nums[0], nums[0]
        for i in range(1, n):
            temp = [nums[i], maxdp*nums[i], mindp*nums[i]]
            mindp = min(temp)
            maxdp = max(temp)
            ans = max(maxdp, ans)
        return ans
```



#### 01背包

背包最大重量为4。

物品为：

|       | 重量 | 价值 |
| ----- | ---- | ---- |
| 物品0 | 1    | 15   |
| 物品1 | 3    | 20   |
| 物品2 | 4    | 30   |

问背包能背的物品最大价值是多少？

```python
# 二维dp
def test_2_wei_bag_problem1(bag_size, weight, value) -> int: 
	rows, cols = len(weight), bag_size + 1
	dp = [[0 for _ in range(cols)] for _ in range(rows)]
    # dp[i][j] 表示从下标为[0-i]的物品里任意取，放进容量为j的背包，价值总和最大是多少
	res = 0
    
	# 初始化dp数组. 
	for i in range(rows): 
		dp[i][0] = 0
	first_item_weight, first_item_value = weight[0], value[0]
	for j in range(1, cols): 	
		if first_item_weight <= j: 
			dp[0][j] = first_item_value

	# 更新dp数组: 先遍历物品, 再遍历背包. 
	for i in range(1, len(weight)): 
		cur_weight, cur_val = weight[i], value[i]
		for j in range(1, cols): 
			if cur_weight > j: # 说明背包装不下当前物品. 
				dp[i][j] = dp[i - 1][j] # 所以不装当前物品. 
			else: 
				# 定义dp数组: dp[i][j] 前i个物品里，放进容量为j的背包，价值总和最大是多少。
				dp[i][j] = max(dp[i - 1][j], dp[i - 1][j - cur_weight]+ cur_val)
				if dp[i][j] > res: 
					res = dp[i][j]

	print(dp)


if __name__ == "__main__": 
	bag_size = 4
	weight = [1, 3, 4]
	value = [15, 20, 30]
	test_2_wei_bag_problem1(bag_size, weight, value)
```

```python
# 一维dp
def test_1_wei_bag_problem():
    weight = [1, 3, 4]
    value = [15, 20, 30]
    bag_weight = 4
    # 初始化: 全为0
    dp = [0] * (bag_weight + 1)

    # 必须先遍历物品, 再遍历背包容量
    for i in range(len(weight)):
        # 这里必须倒序,区别二维,因为二维dp保存了i的状态
        for j in range(bag_weight, weight[i] - 1, -1):
            # 递归公式
            dp[j] = max(dp[j], dp[j - weight[i]] + value[i])

    print(dp)

test_1_wei_bag_problem()
```

![416.分割等和子集1](https://camo.githubusercontent.com/5c5af3f54a3503cdb989ab1c28e2933202a33259608c70af0e72db5a858f14e6/68747470733a2f2f696d672d626c6f672e6373646e696d672e636e2f32303231303131373137313330373430372e706e67)

01背包是每个物品都是1个

完全背包是每个物品无数个

多重背包是每个物品，数量不同的情况

对于leetcode掌握01背包和完全背包足矣

#### 416. Partition Equal Subset Sum

Given a **non-empty** array `nums` containing **only positive integers**, find if the array can be partitioned into two subsets such that the sum of elements in both subsets is equal.

 

**Example 1:**

```
Input: nums = [1,5,11,5]
Output: true
Explanation: The array can be partitioned as [1, 5, 5] and [11].
```

**Example 2:**

```
Input: nums = [1,2,3,5]
Output: false
Explanation: The array cannot be partitioned into equal sum subsets.
```

```python
class Solution:
    def canPartition(self, nums: List[int]) -> bool:
        target = sum(nums)
        if target % 2 == 1: return False
        target //= 2
        # 套到本题，dp[i]表示 背包总容量是i，最大可以凑成i的子集总和为dp[i]
        dp = [0] * target
        # 本题，相当于背包里放入数值，那么物品i的重量是nums[i]，其价值也是nums[i]
        # 所以递推公式：dp[j] = max(dp[j], dp[j - nums[i]] + nums[i]);
        for i in range(len(nums)):
            for j in range(target, nums[i] - 1, -1):
                dp[j] = max(dp[j], dp[j - nums[i]] + nums[i])
        return target == dp[target]
```



#### 1049. Last Stone Weight II

You are given an array of integers `stones` where `stones[i]` is the weight of the `ith` stone.

We are playing a game with the stones. On each turn, we choose any two stones and smash them together. Suppose the stones have weights `x` and `y` with `x <= y`. The result of this smash is:

- If `x == y`, both stones are destroyed, and
- If `x != y`, the stone of weight `x` is destroyed, and the stone of weight `y` has new weight `y - x`.

At the end of the game, there is **at most one** stone left.

Return *the smallest possible weight of the left stone*. If there are no stones left, return `0`.

 

**Example 1:**

```
Input: stones = [2,7,4,1,8,1]
Output: 1
Explanation:
We can combine 2 and 4 to get 2, so the array converts to [2,7,1,8,1] then,
we can combine 7 and 8 to get 1, so the array converts to [2,1,1,1] then,
we can combine 2 and 1 to get 1, so the array converts to [1,1,1] then,
we can combine 1 and 1 to get 0, so the array converts to [1], then that's the optimal value.
```

**Example 2:**

```
Input: stones = [31,26,33,21,40]
Output: 5
```

**Example 3:**

```
Input: stones = [1,2]
Output: 1
```

```python
class Solution:
    def lastStoneWeightII(self, stones: List[int]) -> int:
        # 本题其实就是尽量让石头分成重量相同的两堆，相撞之后剩下的石头最小
        # 是不是感觉和昨天讲解的416. 分割等和子集非常像了
        # dp[j]表示容量（这里说容量更形象，其实就是重量）为j的背包，最多可以背dp[j]这么重的石头
        total = sum(stones)
        half = total // 2
        dp = [0] * (half + 1)
        
        for i in range(len(stones)):
            for j in range(half, stones[i] - 1, -1):
                dp[j] = max(dp[j], dp[j - stones[i]] + stones[i])
        # 最后dp[target]里是容量为target的背包所能背的最大重量
        # 那么分成两堆石头，一堆石头的总重量是dp[target]，另一堆就是sum - dp[target]
        # target = sum / 2 因为是向下取整，所以sum - dp[target] 一定是大于等于dp[target]的
        # 那么相撞之后剩下的最小石头重量就是 (sum - dp[target]) - dp[target]
        return (total - dp[half]) - dp[half]
```

#### 494. Target Sum

You are given an integer array `nums` and an integer `target`.

You want to build an **expression** out of nums by adding one of the symbols `'+'` and `'-'` before each integer in nums and then concatenate all the integers.

- For example, if `nums = [2, 1]`, you can add a `'+'` before `2` and a `'-'` before `1` and concatenate them to build the expression `"+2-1"`.

Return the number of different **expressions** that you can build, which evaluates to `target`.

 

**Example 1:**

```
Input: nums = [1,1,1,1,1], target = 3
Output: 5
Explanation: There are 5 ways to assign symbols to make the sum of nums be target 3.
-1 + 1 + 1 + 1 + 1 = 3
+1 - 1 + 1 + 1 + 1 = 3
+1 + 1 - 1 + 1 + 1 = 3
+1 + 1 + 1 - 1 + 1 = 3
+1 + 1 + 1 + 1 - 1 = 3
```

**Example 2:**

```
Input: nums = [1], target = 1
Output: 1
```

```python
class Solution:
    def findTargetSumWays(self, nums: List[int], target: int) -> int:
        sumValue = sum(nums)
        if target > sumValue or (sumValue + target) % 2 == 1: return 0
        bagSize = (sumValue + target) // 2
        dp = [0] * (bagSize + 1)
        # 假设加法的总和为x，那么减法对应的总和就是sum - x。
        # 所以我们要求的是 x - (sum - x) = S
        # x = (S + sum) / 2
        # 此时问题就转化为，装满容量为x背包，有几种方法。
        # dp[j] 表示：填满j（包括j）这么大容积的包，有dp[i]种方法
        dp[0] = 1
        for i in range(len(nums)):
            for j in range(bagSize, nums[i] - 1, -1):
                dp[j] += dp[j - nums[i]]
                # 在求装满背包有几种方法的情况下，递推公式一般为：
                # dp[j] += dp[j - nums[i]];
        return dp[bagSize]
```

#### 474. Ones and Zeroes

You are given an array of binary strings `strs` and two integers `m` and `n`.

Return *the size of the largest subset of `strs` such that there are **at most*** `m` `0`*'s and* `n` `1`*'s in the subset*.

A set `x` is a **subset** of a set `y` if all elements of `x` are also elements of `y`.

 

**Example 1:**

```
Input: strs = ["10","0001","111001","1","0"], m = 5, n = 3
Output: 4
Explanation: The largest subset with at most 5 0's and 3 1's is {"10", "0001", "1", "0"}, so the answer is 4.
Other valid but smaller subsets include {"0001", "1"} and {"10", "1", "0"}.
{"111001"} is an invalid subset because it contains 4 1's, greater than the maximum of 3.
```

**Example 2:**

```
Input: strs = ["10","0","1"], m = 1, n = 1
Output: 2
Explanation: The largest subset is {"0", "1"}, so the answer is 2.
```

```python
# m 和 n相当于是一个背包，两个维度的背包
# 背包有两个维度，一个是m 一个是n，而不同长度的字符串就是不同大小的待装物品
# dp[i][j]：最多有i个0和j个1的strs的最大子集的大小为dp[i][j]
# dp[i][j] 可以由前一个strs里的字符串推导出来，strs里的字符串有zeroNum个0，oneNum个1
# dp[i][j] 就可以是 dp[i - zeroNum][j - oneNum] + 1
class Solution:
    def findMaxForm(self, strs: List[str], m: int, n: int) -> int:
        dp = [[0] * (n + 1) for _ in range(m + 1)]	# 默认初始化0
        # 先遍历物品
        for str in strs:
            ones = str.count('1')
            zeros = str.count('0')
            # 遍历背包容量且从后向前遍历！
            # 这两层for循环先后循序没有什么讲究，都是物品重量的一个维度，先遍历那个都行
            for i in range(m, zeros - 1, -1):
                for j in range(n, ones - 1, -1):
                    dp[i][j] = max(dp[i][j], dp[i - zeros][j - ones] + 1)
        return dp[m][n]
```

#### 完全背包

```python
# 先遍历物品，再遍历背包
def test_complete_pack1():
    weight = [1, 3, 4]
    value = [15, 20, 30]
    bag_weight = 4

    dp = [0]*(bag_weight + 1)

    for i in range(len(weight)):
        for j in range(weight[i], bag_weight + 1):
            dp[j] = max(dp[j], dp[j - weight[i]] + value[i])
    
    print(dp[bag_weight])

# 先遍历背包，再遍历物品
def test_complete_pack2():
    weight = [1, 3, 4]
    value = [15, 20, 30]
    bag_weight = 4

    dp = [0]*(bag_weight + 1)

    for j in range(bag_weight + 1):
        for i in range(len(weight)):
            if j >= weight[i]: dp[j] = max(dp[j], dp[j - weight[i]] + value[i])
    
    print(dp[bag_weight])


if __name__ == '__main__':
    test_complete_pack1()
    test_complete_pack2()
```



#### 518. Coin Change 2

You are given an integer array `coins` representing coins of different denominations and an integer `amount` representing a total amount of money.

Return *the number of combinations that make up that amount*. If that amount of money cannot be made up by any combination of the coins, return `0`.

You may assume that you have an infinite number of each kind of coin.

The answer is **guaranteed** to fit into a signed **32-bit** integer.

 

**Example 1:**

```
Input: amount = 5, coins = [1,2,5]
Output: 4
Explanation: there are four ways to make up the amount:
5=5
5=2+2+1
5=2+1+1+1
5=1+1+1+1+1
```

**Example 2:**

```
Input: amount = 3, coins = [2]
Output: 0
Explanation: the amount of 3 cannot be made up just with coins of 2.
```

**Example 3:**

```
Input: amount = 10, coins = [10]
Output: 1
```

```python
# 如果求组合数就是外层for循环遍历物品，内层for遍历背包。
# 如果求排列数就是外层for遍历背包，内层for循环遍历物品。
class Solution:
    def change(self, amount: int, coins: List[int]) -> int:
        dp = [0]*(amount + 1)
        dp[0] = 1
        # 组合问题
        # dp[j]：凑成总金额j的货币组合数为dp[j]
        for i in range(len(coins)): # 遍历物品
            for j in range(coins[i], amount + 1): # 遍历背包
                dp[j] += dp[j - coins[i]]
                # 求装满背包有几种方法，一般公式都是：dp[j] += dp[j - nums[i]];
        return dp[amount]
```



#### 377. Combination Sum IV

Given an array of **distinct** integers `nums` and a target integer `target`, return *the number of possible combinations that add up to* `target`.

The answer is **guaranteed** to fit in a **32-bit** integer.

 

**Example 1:**

```
Input: nums = [1,2,3], target = 4
Output: 7
Explanation:
The possible combination ways are:
(1, 1, 1, 1)
(1, 1, 2)
(1, 2, 1)
(1, 3)
(2, 1, 1)
(2, 2)
(3, 1)
Note that different sequences are counted as different combinations.
```

**Example 2:**

```
Input: nums = [9], target = 3
Output: 0
```

```python
# 如果求组合数就是外层for循环遍历物品，内层for遍历背包。
# 如果求排列数就是外层for遍历背包，内层for循环遍历物品。
class Solution:
    def combinationSum4(self, nums, target):
        dp = [0] * (target + 1)
        dp[0] = 1
        # 排列问题
        # dp[i]: 凑成目标正整数为i的排列个数为dp[i]
        for i in range(1, target+1): # 遍历背包
            for j in nums: # 遍历物品
                if i >= j:
                    dp[i] += dp[i - j]
        return dp[-1]
```



#### 322. Coin Change

You are given an integer array `coins` representing coins of different denominations and an integer `amount` representing a total amount of money.

Return *the fewest number of coins that you need to make up that amount*. If that amount of money cannot be made up by any combination of the coins, return `-1`.

You may assume that you have an infinite number of each kind of coin.

 

**Example 1:**

```
Input: coins = [1,2,5], amount = 11
Output: 3
Explanation: 11 = 5 + 5 + 1
```

**Example 2:**

```
Input: coins = [2], amount = 3
Output: -1
```

```python
class Solution:
    # dp[j]：凑足总额为j所需钱币的最少个数为dp[j]
    def coinChange(self, coins: List[int], amount: int) -> int:
        '''版本一'''
        # 初始化
        dp = [amount + 1]*(amount + 1)
        dp[0] = 0
        # 遍历物品
        for coin in coins:
            # 遍历背包
            for j in range(coin, amount + 1):
                dp[j] = min(dp[j], dp[j - coin] + 1)
        return dp[amount] if dp[amount] < amount + 1 else -1
    
    def coinChange1(self, coins: List[int], amount: int) -> int:
        '''版本二'''
        # 初始化
        dp = [amount + 1]*(amount + 1)
        dp[0] = 0
        # 遍历背包
        for j in range(1, amount + 1):
            # 遍历物品
            for coin in coins:
                if j >= coin:
                	dp[j] = min(dp[j], dp[j - coin] + 1)
        return dp[amount] if dp[amount] < amount + 1 else -1

```



#### 279. Perfect Squares

Given an integer `n`, return *the least number of perfect square numbers that sum to* `n`.

A **perfect square** is an integer that is the square of an integer; in other words, it is the product of some integer with itself. For example, `1`, `4`, `9`, and `16` are perfect squares while `3` and `11` are not.

 

**Example 1:**

```
Input: n = 12
Output: 3
Explanation: 12 = 4 + 4 + 4.
```

**Example 2:**

```
Input: n = 13
Output: 2
Explanation: 13 = 4 + 9.
```

```python
class Solution:
    def numSquares(self, n: int) -> int:
        '''版本一'''
        # 初始化
        nums = [i**2 for i in range(1, n + 1) if i**2 <= n]
        dp = [i for i in range(n + 1)]

        # 遍历背包
        for j in range(1, n + 1):
            # 遍历物品
            for num in nums:
                if j >= num:
                    dp[j] = min(dp[j], dp[j - num] + 1)
        return dp[n]
    
    def numSquares1(self, n: int) -> int:
        '''版本二'''
        # 初始化
        nums = [i**2 for i in range(1, n + 1) if i**2 <= n]
        dp = [i for i in range(n + 1)]
        # 遍历物品
        for num in nums:
            # 遍历背包
            for j in range(num, n + 1)
                dp[j] = min(dp[j], dp[j - num] + 1)
        return dp[n]

```



#### 139. Word Break

Given a string `s` and a dictionary of strings `wordDict`, return `true` if `s` can be segmented into a space-separated sequence of one or more dictionary words.

**Note** that the same word in the dictionary may be reused multiple times in the segmentation.

 

**Example 1:**

```
Input: s = "leetcode", wordDict = ["leet","code"]
Output: true
Explanation: Return true because "leetcode" can be segmented as "leet code".
```

**Example 2:**

```
Input: s = "applepenapple", wordDict = ["apple","pen"]
Output: true
Explanation: Return true because "applepenapple" can be segmented as "apple pen apple".
Note that you are allowed to reuse a dictionary word.
```

**Example 3:**

```
Input: s = "catsandog", wordDict = ["cats","dog","sand","and","cat"]
Output: false
```

```python
class Solution:
    def wordBreak(self, s: str, wordDict: List[str]) -> bool:
        # 字符串分割其实是一个排列问题（而不是组合）（apple, pen apple）和(apple, apple, pen)不一样
        dp = [False]*(len(s) + 1)
        # dp[i] : 字符串长度为i的话，dp[i]为true，表示可以拆分为一个或多个在字典中出现的单词。
        dp[0] = True
        for j in range(1, len(s) + 1): # 遍历背包
            for word in wordDict: # 遍历单词
                if j >= len(word):
                    dp[j] = dp[j] or (dp[j - len(word)] and word == s[j - len(word):j])
                    print(dp)
        return dp[len(s)]
```

```python
class Solution:
    def wordBreak(self, s: str, wordDict: List[str]) -> bool:       
        n=len(s)
        dp=[False]*(n+1)
        dp[0]=True
        for i in range(n):
            for j in range(i+1,n+1):
                if(dp[i] and (s[i:j] in wordDict)):
                    dp[j]=True
        return dp[-1]
```

```python
# 回溯
class Solution:
    def wordBreak(self, s: str, wordDict: List[str]) -> bool:
        import functools
        @functools.lru_cache(None)
        def back_track(s):
            if(not s):
                return True
            res=False
            for i in range(1,len(s)+1):
                if(s[:i] in wordDict):
                    res=back_track(s[i:]) or res
            return res
        return back_track(s)
```

#### 198. House Robber

ou are a professional robber planning to rob houses along a street. Each house has a certain amount of money stashed, the only constraint stopping you from robbing each of them is that adjacent houses have security systems connected and **it will automatically contact the police if two adjacent houses were broken into on the same night**.

Given an integer array `nums` representing the amount of money of each house, return *the maximum amount of money you can rob tonight **without alerting the police***.

 

**Example 1:**

```
Input: nums = [1,2,3,1]
Output: 4
Explanation: Rob house 1 (money = 1) and then rob house 3 (money = 3).
Total amount you can rob = 1 + 3 = 4.
```

```python
class Solution:
    def rob(self, nums: List[int]) -> int:
        if len(nums) == 0:
            return 0
        if len(nums) == 1:
            return nums[0]
        # dp[i]：考虑下标i（包括i）以内的房屋，最多可以偷窃的金额为dp[i]
        dp = [0] * len(nums)
        dp[0] = nums[0]
        dp[1] = max(nums[0], nums[1])
        for i in range(2, len(nums)):
            dp[i] = max(dp[i-2]+nums[i], dp[i-1])
        return dp[-1]
```

#### 213. House Robber II

You are a professional robber planning to rob houses along a street. Each house has a certain amount of money stashed. All houses at this place are **arranged in a circle.** That means the first house is the neighbor of the last one. Meanwhile, adjacent houses have a security system connected, and **it will automatically contact the police if two adjacent houses were broken into on the same night**.

Given an integer array `nums` representing the amount of money of each house, return *the maximum amount of money you can rob tonight **without alerting the police***.

 

**Example 1:**

```
Input: nums = [2,3,2]
Output: 3
Explanation: You cannot rob house 1 (money = 2) and then rob house 3 (money = 2), because they are adjacent houses.
```

**Example 2:**

```
Input: nums = [1,2,3,1]
Output: 4
Explanation: Rob house 1 (money = 1) and then rob house 3 (money = 3).
Total amount you can rob = 1 + 3 = 4.
```

**Example 3:**

```
Input: nums = [0]
Output: 0
```

```python
class Solution:
    def rob(self, nums: List[int]) -> int:
        if (n := len(nums)) == 0:
            return 0
        if n == 1:
            return nums[0]
        result1 = self.robRange(nums, 0, n - 2)
        result2 = self.robRange(nums, 1, n - 1)
        return max(result1 , result2)

    def robRange(self, nums: List[int], start: int, end: int) -> int:
        if end == start: return nums[start]
        dp = [0] * len(nums)
        dp[start] = nums[start]
        dp[start + 1] = max(nums[start], nums[start + 1])
        for i in range(start + 2, end + 1):
            dp[i] = max(dp[i -2] + nums[i], dp[i - 1])
        return dp[end]
```



#### 337. House Robber III

The thief has found himself a new place for his thievery again. There is only one entrance to this area, called `root`.

Besides the `root`, each house has one and only one parent house. After a tour, the smart thief realized that all houses in this place form a binary tree. It will automatically contact the police if **two directly-linked houses were broken into on the same night**.

Given the `root` of the binary tree, return *the maximum amount of money the thief can rob **without alerting the police***.

 

**Example 1:**

![img](https://assets.leetcode.com/uploads/2021/03/10/rob1-tree.jpg)

```
Input: root = [3,2,3,null,3,null,1]
Output: 7
Explanation: Maximum amount of money the thief can rob = 3 + 3 + 1 = 7.
```

```python
# 暴力递归

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def rob(self, root: TreeNode) -> int:
        if root is None:
            return 0
        if root.left is None and root.right  is None:
            return root.val
        # 偷父节点
        val1 = root.val
        if root.left:
            val1 += self.rob(root.left.left) + self.rob(root.left.right)
        if root.right:
            val1 += self.rob(root.right.left) + self.rob(root.right.right)
        # 不偷父节点
        val2 = self.rob(root.left) + self.rob(root.right)
        return max(val1, val2)
```

```python
# 记忆化递归

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    memory = {}
    def rob(self, root: TreeNode) -> int:
        if root is None:
            return 0
        if root.left is None and root.right  is None:
            return root.val
        if self.memory.get(root) is not None:
            return self.memory[root]
        # 偷父节点
        val1 = root.val
        if root.left:
            val1 += self.rob(root.left.left) + self.rob(root.left.right)
        if root.right:
            val1 += self.rob(root.right.left) + self.rob(root.right.right)
        # 不偷父节点
        val2 = self.rob(root.left) + self.rob(root.right)
        self.memory[root] = max(val1, val2)
        return max(val1, val2)
```

```python
# DP
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def rob(self, root: TreeNode) -> int:
        result = self.rob_tree(root)
        return max(result[0], result[1])
    
    def rob_tree(self, node):
        if node is None:
            return (0, 0) # (偷当前节点金额，不偷当前节点金额)
        left = self.rob_tree(node.left)
        right = self.rob_tree(node.right)
        val1 = node.val + left[1] + right[1] # 偷当前节点，不能偷子节点
        val2 = max(left[0], left[1]) + max(right[0], right[1]) # 不偷当前节点，可偷可不偷子节点
        return (val1, val2)
```



#### 121. Best Time to Buy and Sell Stock

You are given an array `prices` where `prices[i]` is the price of a given stock on the `ith` day.

You want to maximize your profit by choosing a **single day** to buy one stock and choosing a **different day in the future** to sell that stock.

Return *the maximum profit you can achieve from this transaction*. If you cannot achieve any profit, return `0`.

**Example 1:**

```
Input: prices = [7,1,5,3,6,4]
Output: 5
Explanation: Buy on day 2 (price = 1) and sell on day 5 (price = 6), profit = 6-1 = 5.
Note that buying on day 2 and selling on day 1 is not allowed because you must buy before you sell.
```

```python
# 贪心
def maxProfit(self, prices: List[int]) -> int:
    max_profi, min_price = 0, float("inf")
    for i in prices:
        min_price = min(min_price, i)
        max_profi = max(max_profi, i - min_price)
    return max_profi
```

```python
# DP
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        length = len(prices)
        if len == 0:
            return 0
        dp = [[0] * 2 for _ in range(length)]
        # dp[i][0] 表示第i天持有股票所得最多现金
        # dp[i][1] 表示第i天不持有股票所得最多现金
        dp[0][0] = -prices[0]
        dp[0][1] = 0
        for i in range(1, length):
            dp[i][0] = max(dp[i-1][0], -prices[i])
            dp[i][1] = max(dp[i-1][1], dp[i-1][0] + prices[i])
        return dp[-1][1]
```



#### 122. Best Time to Buy and Sell Stock II

You are given an array `prices` where `prices[i]` is the price of a given stock on the `ith` day.

Find the maximum profit you can achieve. You may complete as many transactions as you like (i.e., buy one and sell one share of the stock multiple times).

**Note:** You may not engage in multiple transactions simultaneously (i.e., you must sell the stock before you buy again).

**Example 1:**

```
Input: prices = [7,1,5,3,6,4]
Output: 7
Explanation: Buy on day 2 (price = 1) and sell on day 3 (price = 5), profit = 5-1 = 4.
Then buy on day 4 (price = 3) and sell on day 5 (price = 6), profit = 6-3 = 3.
```

```python
def maxProfit(self, prices: List[int]) -> int:
    profit = 0
    for i in range(len(prices)-1):
        if prices[i+1] > prices[i]: profit += prices[i+1] - prices[i]
    return profit
```

```python
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        length = len(prices)
        dp = [[0] * 2 for _ in range(length)]
        dp[0][0] = -prices[0]
        dp[0][1] = 0
        for i in range(1, length):
            dp[i][0] = max(dp[i-1][0], dp[i-1][1] - prices[i]) 
            #注意这里是和121. 买卖股票的最佳时机唯一不同的地方
            dp[i][1] = max(dp[i-1][1], dp[i-1][0] + prices[i])
        return dp[-1][1]

```

#### 123. Best Time to Buy and Sell Stock III

You are given an array `prices` where `prices[i]` is the price of a given stock on the `ith` day.

Find the maximum profit you can achieve. You may complete **at most two transactions**.

**Note:** You may not engage in multiple transactions simultaneously (i.e., you must sell the stock before you buy again).

 

**Example 1:**

```
Input: prices = [3,3,5,0,0,3,1,4]
Output: 6
Explanation: Buy on day 4 (price = 0) and sell on day 6 (price = 3), profit = 3-0 = 3.
Then buy on day 7 (price = 1) and sell on day 8 (price = 4), profit = 4-1 = 3.
```

```python
class Solution:
    # 一天一共就有五个状态， 
    # 0.没有操作
    # 1.第一次买入
    # 2.第一次卖出
    # 3.第二次买入
    # 4.第二次卖出
    # dp[i][j]中 i表示第i天，j为 [0 - 4] 五个状态，dp[i][j]表示第i天状态j所剩最大现金。
    def maxProfit(self, prices: List[int]) -> int:
        if len(prices) == 0:
            return 0
        dp = [[0] * 5 for _ in range(len(prices))]
        dp[0][1] = -prices[0]
        dp[0][3] = -prices[0]
        for i in range(1, len(prices)):
            dp[i][0] = dp[i-1][0]
            dp[i][1] = max(dp[i-1][1], dp[i-1][0] - prices[i])
            dp[i][2] = max(dp[i-1][2], dp[i-1][1] + prices[i])
            dp[i][3] = max(dp[i-1][3], dp[i-1][2] - prices[i])
            dp[i][4] = max(dp[i-1][4], dp[i-1][3] + prices[i])
        return dp[-1][4]
```

#### 188. Best Time to Buy and Sell Stock IV

You are given an integer array `prices` where `prices[i]` is the price of a given stock on the `ith` day, and an integer `k`.

Find the maximum profit you can achieve. You may complete at most `k` transactions.

**Note:** You may not engage in multiple transactions simultaneously (i.e., you must sell the stock before you buy again).

 

**Example 1:**

```
Input: k = 2, prices = [2,4,1]
Output: 2
Explanation: Buy on day 1 (price = 2) and sell on day 2 (price = 4), profit = 4-2 = 2.
```

```python
# 使用二维数组 dp[i][j] ：第i天的状态为j，所剩下的最大现金是dp[i][j]
# j的状态表示为：
# 0 表示不操作
# 1 第一次买入
# 2 第一次卖出
# 3 第二次买入
# 4 第二次卖出
# .....
# 大家应该发现规律了吧 ，除了0以外，偶数就是卖出，奇数就是买入。
# 题目要求是至多有K笔交易，那么j的范围就定义为 2 * k + 1 就可以了。
class Solution:
    def maxProfit(self, k: int, prices: List[int]) -> int:
        if len(prices) == 0:
            return 0
        dp = [[0] * (2*k+1) for _ in range(len(prices))]
        for j in range(1, 2*k, 2):
            dp[0][j] = -prices[0]
        for i in range(1, len(prices)):
            for j in range(0, 2*k-1, 2):
                dp[i][j+1] = max(dp[i-1][j+1], dp[i-1][j] - prices[i])
                dp[i][j+2] = max(dp[i-1][j+2], dp[i-1][j+1] + prices[i])
        return dp[-1][2*k]
```

#### 309. Best Time to Buy and Sell Stock with Cooldown

You are given an array `prices` where `prices[i]` is the price of a given stock on the `ith` day.

Find the maximum profit you can achieve. You may complete as many transactions as you like (i.e., buy one and sell one share of the stock multiple times) with the following restrictions:

- After you sell your stock, you cannot buy stock on the next day (i.e., cooldown one day).

**Note:** You may not engage in multiple transactions simultaneously (i.e., you must sell the stock before you buy again).

 

**Example 1:**

```
Input: prices = [1,2,3,0,2]
Output: 3
Explanation: transactions = [buy, sell, cooldown, buy, sell]
```

**Example 2:**

```
Input: prices = [1]
Output: 0
```

```python
# dp[i][j]，第i天状态为j，所剩的最多现金为dp[i][j]。
# 具体可以区分出如下四个状态：

# 状态0：买入股票状态（今天买入股票，或者是之前就买入了股票然后没有操作）
# 卖出股票状态，这里就有两种卖出股票状态
# 	状态1：两天前就卖出了股票，度过了冷冻期，一直没操作，今天保持卖出股票状态
# 	状态2：今天卖出了股票
# 状态3：今天为冷冻期状态，但冷冻期状态不可持续，只有一天！
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        n = len(prices)
        if n == 0:
            return 0
        dp = [[0] * 4 for _ in range(n)]
        dp[0][0] = -prices[0] #持股票
        for i in range(1, n):
            dp[i][0] = max(dp[i-1][0], max(dp[i-1][3], dp[i-1][1]) - prices[i])
            dp[i][1] = max(dp[i-1][1], dp[i-1][3]) # 不能是昨天卖出，因为昨天卖出则今天是状态3
            dp[i][2] = dp[i-1][0] + prices[i]
            dp[i][3] = dp[i-1][2]
        return max(dp[n-1][3], dp[n-1][1], dp[n-1][2])
```



#### 714. Best Time to Buy and Sell Stock with Transaction Fee

You are given an array `prices` where `prices[i]` is the price of a given stock on the `ith` day, and an integer `fee` representing a transaction fee.

Find the maximum profit you can achieve. You may complete as many transactions as you like, but you need to pay the transaction fee for each transaction.

**Note:** You may not engage in multiple transactions simultaneously (i.e., you must sell the stock before you buy again).

 

**Example 1:**

```
Input: prices = [1,3,2,8,4,9], fee = 2
Output: 8
Explanation: The maximum profit can be achieved by:
- Buying at prices[0] = 1
- Selling at prices[3] = 8
- Buying at prices[4] = 4
- Selling at prices[5] = 9
The total profit is ((8 - 1) - 2) + ((9 - 4) - 2) = 8.
```

```python
class Solution:
    def maxProfit(self, prices: List[int], fee: int) -> int:
        profit = 0
        buy = prices[0]
        for i in range(1, len(prices)):
            if prices[i] < buy:
                buy = prices[i]
            elif prices[i] >= buy and prices[i] - buy <= fee:
                continue
            else:
                profit += prices[i] - buy - fee
                buy = prices[i] - fee
        return profit
```

```python
class Solution:
    def maxProfit(self, prices: List[int], fee: int) -> int:
        n = len(prices)
        dp = [[0] * 2 for _ in range(n)]
        # 0 : 持股（买入）
        # 1 : 不持股（售出）
        # dp 定义第i天持股/不持股 所得最多现金
        dp[0][0] = -prices[0] #初始化持股
        for i in range(1, n):
            dp[i][0] = max(dp[i-1][0], dp[i-1][1] - prices[i])
            dp[i][1] = max(dp[i-1][1], dp[i-1][0] + prices[i] - fee)
        return max(dp[-1][0], dp[-1][1])
```





#### 300. Longest Increasing Subsequence

Given an integer array `nums`, return the length of the longest strictly increasing subsequence.

A **subsequence** is a sequence that can be derived from an array by deleting some or no elements without changing the order of the remaining elements. For example, `[3,6,2,7]` is a subsequence of the array `[0,3,1,6,2,2,7]`.

 

**Example 1:**

```
Input: nums = [10,9,2,5,3,7,101,18]
Output: 4
Explanation: The longest increasing subsequence is [2,3,7,101], therefore the length is 4.
```

**Example 2:**

```
Input: nums = [0,1,0,3,2,3]
Output: 4
```

**Example 3:**

```
Input: nums = [7,7,7,7,7,7,7]
Output: 1
```

```python
class Solution:
    def lengthOfLIS(self, nums: List[int]) -> int:
        if len(nums) <= 1:
            return len(nums)
        dp = [1] * len(nums)
        # dp[i]表示i之前包括i的最长上升子序列
        result = 0
        for i in range(1, len(nums)):
            for j in range(0, i):
                if nums[i] > nums[j]:
                    dp[i] = max(dp[i], dp[j] + 1)
                result = max(result, dp[i])
        return result
```

#### 674. Longest Continuous Increasing Subsequence

Given an unsorted array of integers `nums`, return *the length of the longest **continuous increasing subsequence** (i.e. subarray)*. The subsequence must be **strictly** increasing.

A **continuous increasing subsequence** is defined by two indices `l` and `r` (`l < r`) such that it is `[nums[l], nums[l + 1], ..., nums[r - 1], nums[r]]` and for each `l <= i < r`, `nums[i] < nums[i + 1]`.

 

**Example 1:**

```
Input: nums = [1,3,5,4,7]
Output: 3
Explanation: The longest continuous increasing subsequence is [1,3,5] with length 3.
Even though [1,3,5,7] is an increasing subsequence, it is not continuous as elements 5 and 7 are separated by element
4.
```

```python
class Solution:
    def findLengthOfLCIS(self, nums: List[int]) -> int:
        if len(nums) == 0:
            return 0
        dp = [1] * len(nums)
        result = 0
        for i in range(len(nums)-1):
            if nums[i+1] > nums[i]: #连续记录
                dp[i+1] = dp[i] + 1
            result = max(result, dp[i+1])
        return result
```

```python
class Solution:
    def findLengthOfLCIS(self, nums: List[int]) -> int:
        if len(nums) == 0:
            return 0
        result = 1 #连续子序列最少也是1
        count = 1
        for i in range(len(nums)-1):
            if nums[i+1] > nums[i]: #连续记录
                count += 1
            else: #不连续，count从头开始
                count = 1
            result = max(result, count)
        return result
```

#### 718. Maximum Length of Repeated Subarray

Given two integer arrays `nums1` and `nums2`, return *the maximum length of a subarray that appears in **both** arrays*.

**Example 1:**

```
Input: nums1 = [1,2,3,2,1], nums2 = [3,2,1,4,7]
Output: 3
Explanation: The repeated subarray with maximum length is [3,2,1].
```

```python
class Solution:
    def findLength(self, A: List[int], B: List[int]) -> int:
        dp = [[0] * (len(B)+1) for _ in range(len(A)+1)]
        result = 0
        for i in range(1, len(A)+1):
            for j in range(1, len(B)+1):
                if A[i-1] == B[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                result = max(result, dp[i][j])
        return result
```

```python
class Solution:
    def findLength(self, A: List[int], B: List[int]) -> int:
        dp = [0] * (len(B) + 1)
        result = 0
        for i in range(1, len(A)+1):
            for j in range(len(B), 0, -1):
                if A[i-1] == B[j-1]:
                    dp[j] = dp[j-1] + 1
                else:
                    dp[j] = 0 #注意这里不相等的时候要有赋0的操作
                result = max(result, dp[j])
        return result
```

#### 1143. Longest Common Subsequence

Given two strings `text1` and `text2`, return *the length of their longest **common subsequence**.* If there is no **common subsequence**, return `0`.

A **subsequence** of a string is a new string generated from the original string with some characters (can be none) deleted without changing the relative order of the remaining characters.

- For example, `"ace"` is a subsequence of `"abcde"`.

A **common subsequence** of two strings is a subsequence that is common to both strings.

 

**Example 1:**

```
Input: text1 = "abcde", text2 = "ace" 
Output: 3  
Explanation: The longest common subsequence is "ace" and its length is 3.
```

```python
class Solution:
    def longestCommonSubsequence(self, text1: str, text2: str) -> int:
        len1, len2 = len(text1)+1, len(text2)+1
        dp = [[0 for _ in range(len1)] for _ in range(len2)] 
        # dp[i][j] ：以下标i - 1为结尾的A，和以下标j - 1为结尾的B，最长重复子数组长度为dp[i][j]
        for i in range(1, len2):
            for j in range(1, len1): 
                if text1[j-1] == text2[i-1]:
                    dp[i][j] = dp[i-1][j-1]+1 
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        return dp[-1][-1]
```

#### 1035. Uncrossed Lines

We write the integers of `nums1` and `nums2` (in the order they are given) on two separate horizontal lines.

Now, we may draw *connecting lines*: a straight line connecting two numbers `nums1[i]` and `nums2[j]` such that:

- `nums1[i] == nums2[j]`;
- The line we draw does not intersect any other connecting (non-horizontal) line.

Note that a connecting lines cannot intersect even at the endpoints: each number can only belong to one connecting line.

Return the maximum number of connecting lines we can draw in this way.

 

**Example 1:**

<img src="https://assets.leetcode.com/uploads/2019/04/26/142.png" alt="img" style="zoom:5%;" />

```
Input: nums1 = [1,4,2], nums2 = [1,2,4]
Output: 2
Explanation: We can draw 2 uncrossed lines as in the diagram.
We cannot draw 3 uncrossed lines, because the line from nums1[1]=4 to nums2[2]=4 will intersect the line from nums1[2]=2 to nums2[1]=2.
```

```python
# 本题说是求绘制的最大连线数，其实就是求两个字符串的最长公共子序列的长度
class Solution:
    def maxUncrossedLines(self, A: List[int], B: List[int]) -> int:
        dp = [[0] * (len(B)+1) for _ in range(len(A)+1)]
        for i in range(1, len(A)+1):
            for j in range(1, len(B)+1):
                if A[i-1] == B[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        return dp[-1][-1]
```

#### 53. Maximum Subarray

see 53 in greedy

#### 392. Is Subsequence

Given two strings `s` and `t`, return `true` *if* `s` *is a **subsequence** of* `t`*, or* `false` *otherwise*.

A **subsequence** of a string is a new string that is formed from the original string by deleting some (can be none) of the characters without disturbing the relative positions of the remaining characters. (i.e., `"ace"` is a subsequence of `"abcde"` while `"aec"` is not).

 

**Example 1:**

```
Input: s = "abc", t = "ahbgdc"
Output: true
```

```python
class Solution:
    def isSubsequence(self, s: str, t: str) -> bool:
        n, m = len(s), len(t)
        i = j = 0
        while i < n and j < m:
            if s[i] == t[j]:
                i += 1
            j += 1
        return i == n
```

```python
class Solution:
    def isSubsequence(self, s: str, t: str) -> bool:
        # dp[i][j] 表示以下标i-1为结尾的字符串s，和以下标j-1为结尾的字符串t，相同子序列的长度为dp[i][j]。
        dp = [[0] * (len(t)+1) for _ in range(len(s)+1)]
        for i in range(1, len(s)+1):
            for j in range(1, len(t)+1):
                if s[i-1] == t[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = dp[i][j-1]
        if dp[-1][-1] == len(s):
            return True
        return False
```

#### 91. Decode Ways

A message containing letters from `A-Z` can be **encoded** into numbers using the following mapping:

```
'A' -> "1"
'B' -> "2"
...
'Z' -> "26"
```

To **decode** an encoded message, all the digits must be grouped then mapped back into letters using the reverse of the mapping above (there may be multiple ways). For example, `"11106"` can be mapped into:

- `"AAJF"` with the grouping `(1 1 10 6)`
- `"KJF"` with the grouping `(11 10 6)`

Note that the grouping `(1 11 06)` is invalid because `"06"` cannot be mapped into `'F'` since `"6"` is different from `"06"`.

Given a string `s` containing only digits, return *the **number** of ways to **decode** it*.

The answer is guaranteed to fit in a **32-bit** integer.

 

**Example 1:**

```
Input: s = "12"
Output: 2
Explanation: "12" could be decoded as "AB" (1 2) or "L" (12).
```

```python
class Solution:
    def numDecodings(self, s: str) -> int:
        n = len(s)
        # 设f表示字符串s的前i个字符s[1..i]的解码方法数
        f = [1] + [0] * n
        for i in range(1, n + 1):
            if s[i - 1] != '0':
                f[i] += f[i - 1]
            if i > 1 and s[i - 2] != '0' and int(s[i-2:i]) <= 26:
                f[i] += f[i - 2]
        return f[n]
```

#### 115. Distinct Subsequences



Given two strings `s` and `t`, return *the number of distinct subsequences of `s` which equals `t`*.

A string's **subsequence** is a new string formed from the original string by deleting some (can be none) of the characters without disturbing the remaining characters' relative positions. (i.e., `"ACE"` is a subsequence of `"ABCDE"` while `"AEC"` is not).

It is guaranteed the answer fits on a 32-bit signed integer.

 

**Example 1:**

```
Input: s = "rabbbit", t = "rabbit"
Output: 3
Explanation:
As shown below, there are 3 ways you can generate "rabbit" from S.
rabbbit
rabbbit
rabbbit
```

```python
# 其中 dp[i][j] 表示在 s[i:] 的子序列中 t[j:] 出现的个数
class Solution:
    def numDistinct(self, s: str, t: str) -> int:
        m, n = len(s), len(t)
        if m < n:
            return 0
        
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(m + 1):
            dp[i][n] = 1
        
        for i in range(m - 1, -1, -1):
            for j in range(n - 1, -1, -1):
                if s[i] == t[j]:
                    dp[i][j] = dp[i + 1][j + 1] + dp[i + 1][j]
                else:
                    dp[i][j] = dp[i + 1][j]
        
        return dp[0][0]
```

```python
# dp[i][j]：以i-1为结尾的s子序列中出现以j-1为结尾的t的个数为dp[i][j]
class Solution:
    def numDistinct(self, s: str, t: str) -> int:
        if (m := len(s)) < (n := len(t)):
            return 0
        dp = [[0] * (n+1) for _ in range(m+1)]
        for i in range(m):
            dp[i][0] = 1
        for i in range(1, m+1):
            for j in range(1, n+1):
                if s[i-1] == t[j-1]:
                    dp[i][j] = dp[i-1][j-1] + dp[i-1][j]
                else:
                    dp[i][j] = dp[i-1][j]
        return dp[-1][-1]
    
    
    
class SolutionDP2:
    """
    既然dp[i]只用到dp[i - 1]的状态，
    我们可以通过缓存dp[i - 1]的状态来对dp进行压缩，
    减少空间复杂度。
    （原理等同同于滚动数组）
    """
    def numDistinct(self, s: str, t: str) -> int:
        n1, n2 = len(s), len(t)
        if n1 < n2:
            return 0

        dp = [0 for _ in range(n2 + 1)]
        dp[0] = 1

        for i in range(1, n1 + 1):
            # 必须深拷贝
            # 不然prev[i]和dp[i]是同一个地址的引用
            prev = dp.copy()
            # 剪枝，保证s的长度大于等于t
            # 因为对于任意i，i > n1, dp[i] = 0
            # 没必要跟新状态。 
            end = i if i < n2 else n2
            for j in range(1, end + 1):
                if s[i - 1] == t[j - 1]:
                    dp[j] = prev[j - 1] + prev[j]
                else:
                    dp[j] = prev[j]
        return dp[-1]
```



#### 583. Delete Operation for Two Strings

Given two strings `word1` and `word2`, return *the minimum number of **steps** required to make* `word1` *and* `word2` *the same*.

In one **step**, you can delete exactly one character in either string.

 

**Example 1:**

```
Input: word1 = "sea", word2 = "eat"
Output: 2
Explanation: You need one step to make "sea" to "ea" and another step to make "eat" to "ea".
```

**Example 2:**

```
Input: word1 = "leetcode", word2 = "etco"
Output: 4
```

```python
class Solution:
    def minDistance(self, word1: str, word2: str) -> int:
        # dp[i][j]：以i-1为结尾的字符串word1，和以j-1位结尾的字符串word2，想要达到相等，所需要删除元素的最少次数
        dp = [[0] * (len(word2)+1) for _ in range(len(word1)+1)]
        for i in range(len(word1)+1):
            dp[i][0] = i
        for j in range(len(word2)+1):
            dp[0][j] = j
        for i in range(1, len(word1)+1):
            for j in range(1, len(word2)+1):
                if word1[i-1] == word2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = min(dp[i-1][j-1] + 2, dp[i-1][j] + 1, dp[i][j-1] + 1)
        return dp[-1][-1]
```

#### 72. Edit Distance

Given two strings `word1` and `word2`, return *the minimum number of operations required to convert `word1` to `word2`*.

You have the following three operations permitted on a word:

- Insert a character
- Delete a character
- Replace a character

 

**Example 1:**

```
Input: word1 = "hr", word2 = "ros"
Input: word1 = "hr", word2 = "rosr"
Output: 3
Explanation: 
horse -> rorse (replace 'h' with 'r')
rorse -> rose (remove 'r')
rose -> ros (remove 'e')
```

```python
# dp[i][j] 表示以下标i-1为结尾的字符串word1，和以下标j-1为结尾的字符串word2，最近编辑距离为dp[i][j]
```

在确定递推公式的时候，首先要考虑清楚编辑的几种操作，整理如下：

```
if (word1[i - 1] == word2[j - 1])
    不操作
if (word1[i - 1] != word2[j - 1])
    增
    删
    换
```

也就是如上4种情况。

`if (word1[i - 1] != word2[j - 1])`，此时就需要编辑了，如何编辑呢？

- 操作一：word1增加一个元素，使其word1[i - 1]与word2[j - 1]相同，那么就是以下标i-1为结尾的word1 与 j-2为结尾的word2的最近编辑距离 加上一个增加元素的操作。

即 `dp[i][j] = dp[i][j-1] + 1;`

- 操作二：word2添加一个元素，使其word1[i - 1]与word2[j - 1]相同，那么就是以下标i-2为结尾的word1 与 j-1为结尾的word2的最近编辑距离 加上一个增加元素的操作。

即 `dp[i][j] = dp[i-1][j] + 1;`

这里有同学发现了，怎么都是添加元素，删除元素去哪了。

**word2添加一个元素，相当于word1删除一个元素**，例如 `word1 = "ad" ，word2 = "a"`，`word1`删除元素`'d'`，`word2`添加一个元素`'d'`，变成`word1="a", word2="ad"`， 最终的操作数是一样！ 

其实这里只需要考虑“删除”或者只考虑“插入”，此二者等价。

```python
class Solution:
    def minDistance(self, word1: str, word2: str) -> int:
        dp = [[0] * (len(word2)+1) for _ in range(len(word1)+1)]
        for i in range(len(word1)+1):
            dp[i][0] = i
        for j in range(len(word2)+1):
            dp[0][j] = j
        for i in range(1, len(word1)+1):
            for j in range(1, len(word2)+1):
                if word1[i-1] == word2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = min(dp[i-1][j-1], dp[i-1][j], dp[i][j-1]) + 1
        return dp[-1][-1]
```



#### 647. Palindromic Substrings

Given a string `s`, return *the number of **palindromic substrings** in it*.

A string is a **palindrome** when it reads the same backward as forward.

A **substring** is a contiguous sequence of characters within the string.

 

**Example 1:**

```
Input: s = "abc"
Output: 3
Explanation: Three palindromic strings: "a", "b", "c".
```

```python
# 双指针
class Solution:
    def countSubstrings(self, s: str) -> int:
        result = 0
        for i in range(len(s)):
            result += self.extend(s, i, i, len(s)) #以i为中心
            result += self.extend(s, i, i+1, len(s)) #以i和i+1为中心
        return result
    
    def extend(self, s, i, j, n):
        res = 0
        while i >= 0 and j < n and s[i] == s[j]:
            i -= 1
            j += 1
            res += 1
        return res
```



动规五部曲：

1. 确定dp数组（dp table）以及下标的含义

布尔类型的dp\[i\]\[j\]：表示区间范围[i,j] （注意是左闭右闭）的子串是否是回文子串，如果是dp[i][j]为true，否则为false。

1. 确定递推公式

在确定递推公式时，就要分析如下几种情况。

整体上是两种，就是s[i]与s[j]相等，s[i]与s[j]不相等这两种。

当s[i]与s[j]不相等，那没啥好说的了，dp[i][j]一定是false。

当s[i]与s[j]相等时，这就复杂一些了，有如下三种情况

- 情况一：下标i 与 j相同，同一个字符例如a，当然是回文子串

- 情况二：下标i 与 j相差为1，例如aa，也是文子串

- 情况三：下标：i 与 j相差大于1的时候，例如cabac，此时s[i]与s[j]已经相同了，我们看i到j区间是不是回文子串就看aba是不是回文就可以了，那么aba的区间就是 i+1 与 j-1区间，这个区间是不是回文就看dp\[i + 1\]\[j - 1\]是否为true。

  

  ```python
  class Solution:
      def countSubstrings(self, s: str) -> int:
          dp = [[False] * len(s) for _ in range(len(s))]
          result = 0
          for i in range(len(s)-1, -1, -1): #注意遍历顺序
              for j in range(i, len(s)):
                  if s[i] == s[j]:
                      if j - i <= 1: #情况一 和 情况二
                          result += 1
                          dp[i][j] = True
                      elif dp[i+1][j-1]: #情况三
                          result += 1
                          dp[i][j] = True
          return result
  ```

  #### 516. Longest Palindromic Subsequence

  Given a string `s`, find *the longest palindromic **subsequence**'s length in* `s`.

  A **subsequence** is a sequence that can be derived from another sequence by deleting some or no elements without changing the order of the remaining elements.

   

  **Example 1:**

  ```
  Input: s = "bbbab"
  Output: 4
  Explanation: One possible longest palindromic subsequence is "bbbb".
  ```

动规五部曲分析如下：

1.确定dp数组（dp table）以及下标的含义

dp\[i][j]：字符串s在[i, j]范围内最长的回文子序列的长度为dp\[i][j]。

2.确定递推公式

在判断回文子串的题目中，关键逻辑就是看s[i]与s[j]是否相同。

如果s[i]与s[j]相同，那么dp\[i][j] = dp\[i + 1][j - 1] + 2;

如果s[i]与s[j]不相同，说明s[i]和s[j]的同时加入 并不能增加[i,j]区间回文子串的长度，那么分别加入s[i]、s[j]看看哪一个可以组成最长的回文子序列。

加入s[j]的回文子序列长度为dp\[i + 1][j]。

加入s[i]的回文子序列长度为dp\[i][j - 1]。

那么dp\[i][j]一定是取最大的，即：dp\[i][j] = max(dp\[i + 1][j], dp\[i][j - 1]);

```python
class Solution:
    def longestPalindromeSubseq(self, s: str) -> int:
        dp = [[0] * len(s) for _ in range(len(s))]
        for i in range(len(s)):
            dp[i][i] = 1
        for i in range(len(s)-1, -1, -1):
            for j in range(i+1, len(s)):
                if s[i] == s[j]:
                    dp[i][j] = dp[i+1][j-1] + 2
                else:
                    dp[i][j] = max(dp[i+1][j], dp[i][j-1])
        return dp[0][-1]
```



### Stack and Queue

#### 239. Sliding Window Maximum

You are given an array of integers `nums`, there is a sliding window of size `k` which is moving from the very left of the array to the very right. You can only see the `k` numbers in the window. Each time the sliding window moves right by one position.

Return *the max sliding window*.

 

**Example 1:**

```
Input: nums = [1,3,-1,-3,5,3,6,7], k = 3
Output: [3,3,5,5,6,7]
Explanation: 
Window position                Max
---------------               -----
[1  3  -1] -3  5  3  6  7       3
 1 [3  -1  -3] 5  3  6  7       3
 1  3 [-1  -3  5] 3  6  7       5
 1  3  -1 [-3  5  3] 6  7       5
 1  3  -1  -3 [5  3  6] 7       6
 1  3  -1  -3  5 [3  6  7]      7
```

```python
class Solution:
    def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
        n = len(nums)
        q = collections.deque()
        for i in range(k):
            while q and nums[i] >= nums[q[-1]]: # 单调队列
                q.pop()
            q.append(i)

        ans = [nums[q[0]]]
        for i in range(k, n):
            while q and nums[i] >= nums[q[-1]]:
                q.pop()
            q.append(i)
            while q[0] <= i - k: # 滑动窗口
                q.popleft()
            ans.append(nums[q[0]])
        
        return ans
```

```python
class MyQueue: #单调队列（从大到小
    def __init__(self):
        self.queue = [] #使用list来实现单调队列
    
    #每次弹出的时候，比较当前要弹出的数值是否等于队列出口元素的数值，如果相等则弹出。
    #同时pop之前判断队列当前是否为空。
    def pop(self, value):
        if self.queue and value == self.queue[0]:
            self.queue.pop(0)#list.pop()时间复杂度为O(n),这里可以使用collections.deque()
            
    #如果push的数值大于入口元素的数值，那么就将队列后端的数值弹出，直到push的数值小于等于队列入口元素的数值为止。
    #这样就保持了队列里的数值是单调从大到小的了。
    def push(self, value):
        while self.queue and value > self.queue[-1]:
            self.queue.pop()
        self.queue.append(value)
        
    #查询当前队列里的最大值 直接返回队列前端也就是front就可以了。
    def front(self):
        return self.queue[0]
    
class Solution:
    def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
        que = MyQueue()
        result = []
        for i in range(k): #先将前k的元素放进队列
            que.push(nums[i])
        result.append(que.front()) #result 记录前k的元素的最大值
        for i in range(k, len(nums)):
            que.pop(nums[i - k]) #滑动窗口移除最前面元素
            que.push(nums[i]) #滑动窗口前加入最后面的元素
            result.append(que.front()) #记录对应的最大值
        return result
```

#### 341. Flatten Nested List Iterator

```python
class NestedIterator(object):
    def dfs(self, nests):
        for nest in nests:
            if nest.isInteger():
                self.queue.append(nest.getInteger())
            else:
                self.dfs(nest.getList())
                    
    def __init__(self, nestedList):
        self.queue = collections.deque()
        self.dfs(nestedList)

    def next(self):
        return self.queue.popleft()

    def hasNext(self):
        return len(self.queue)
```

```python
class NestedIterator(object):

    def __init__(self, nestedList):
        self.stack = []
        for i in range(len(nestedList) - 1, -1, -1):
            self.stack.append(nestedList[i])
        

    def next(self):
        cur = self.stack.pop()
        return cur.getInteger()

    def hasNext(self):
        while self.stack:
            cur = self.stack[-1]
            if cur.isInteger():
                return True
            self.stack.pop()
            for i in range(len(cur.getList()) - 1, -1, -1):
                self.stack.append(cur.getList()[i])
        return False
```

#### 单调栈

#### 739. Daily Temperatures

```python
class Solution:
    def dailyTemperatures(self, temperatures: List[int]) -> List[int]:
        n = len(temperatures)
        ans = [0] * n
        stack = []
        
        for i in range(n):
            temp = temperatures[i]
            while stack and temp > temperatures[stack[-1]]:
                index = stack.pop()
                ans[index] = i - index
            stack.append(i)
        
        return ans
```

#### 1996. The Number of Weak Characters in the Game



### Intervals

#### 56. Merge Intervals

Given an array of `intervals` where `intervals[i] = [starti, endi]`, merge all overlapping intervals, and return *an array of the non-overlapping intervals that cover all the intervals in the input*.

 

**Example 1:**

```
Input: intervals = [[1,3],[2,6],[8,10],[15,18]]
Output: [[1,6],[8,10],[15,18]]
Explanation: Since intervals [1,3] and [2,6] overlaps, merge them into [1,6].
```

```python
class Solution:
    def merge(self, intervals: List[List[int]]) -> List[List[int]]:

        intervals.sort(key=lambda x: x[0])

        merged = []
        for interval in intervals:
            # if the list of merged intervals is empty or if the current
            # interval does not overlap with the previous, simply append it.
            if not merged or merged[-1][1] < interval[0]:
                merged.append(interval)
            else:
            # otherwise, there is overlap, so we merge the current and previous
            # intervals.
                merged[-1][1] = max(merged[-1][1], interval[1])

        return merged
```

```python
class Solution:
    def insert(self, intervals: List[List[int]], newInterval: List[int]) -> List[List[int]]:
        intervals.append(newInterval) 
        intervals.sort(key = lambda x: x[0])

        merged = []
        
        for interval in intervals:
            if not merged or merged[-1][1] < interval[0]:
                merged.append(interval)
            else:
                merged[-1][1] = max(merged[-1][1], interval[1])
                
        return merged
```



#### 57. Insert Interval

Given a set of *non-overlapping* intervals, insert a new interval into the intervals (merge if necessary).

You may assume that the intervals were initially sorted according to their start times.

 

**Example 1:**

```
Input: intervals = [[1,3],[6,9]], newInterval = [2,5]
Output: [[1,5],[6,9]]
```

```python
class Solution:
    def insert(self, intervals: List[List[int]], newInterval: List[int]) -> List[List[int]]:
        left, right = newInterval
        placed = False
        ans = list()
        for li, ri in intervals:
            if li > right:
                # 在插入区间的右侧且无交集
                if not placed:
                    ans.append([left, right])
                    placed = True
                ans.append([li, ri])
            elif ri < left:
                # 在插入区间的左侧且无交集
                ans.append([li, ri])
            else:
                # 与插入区间有交集，计算它们的并集
                left = min(left, li)
                right = max(right, ri)
        
        if not placed:
            ans.append([left, right])
        return ans
```





### Strings

#### 6. ZigZag Conversion

The string `"PAYPALISHIRING"` is written in a zigzag pattern on a given number of rows=3 like this: 

```
P   A   H   N
A P L S I I G
Y   I   R
```

And then read line by line: `"PAHNAPLSIIGYIR"`

Write the code that will take a string and make this conversion given a number of rows:

```
string convert(string s, int numRows);
```

 

**Example 1:**

```
Input: s = "PAYPALISHIRING", numRows = 3
Output: "PAHNAPLSIIGYIR"
```

**Example 2:**

```
Input: s = "PAYPALISHIRING", numRows = 4
Output: "PINALSIGYAHRPI"
Explanation:
P     I    N
A   L S  I G
Y A   H R
P     I
```

```python
class Solution:
    def convert(self, s: str, numRows: int) -> str:
        if numRows == 1 or numRows >= len(s):
            return s
        zigzag = ['' for x in range(numRows)]
        row, step = 0, 1
        for ch in s:
            zigzag[row] += ch
            if row == 0: 
                step = 1
            elif row == numRows - 1: 
                step = -1
            row += step
        return ''.join(zigzag)
```



#### 12. Integer to Roman

Roman numerals are represented by seven different symbols: `I`, `V`, `X`, `L`, `C`, `D` and `M`.

```
Symbol       Value
I             1
V             5
X             10
L             50
C             100
D             500
M             1000
```

For example, `2` is written as `II` in Roman numeral, just two one's added together. `12` is written as `XII`, which is simply `X + II`. The number `27` is written as `XXVII`, which is `XX + V + II`.

Roman numerals are usually written largest to smallest from left to right. However, the numeral for four is not `IIII`. Instead, the number four is written as `IV`. Because the one is before the five we subtract it making four. The same principle applies to the number nine, which is written as `IX`. There are six instances where subtraction is used:

- `I` can be placed before `V` (5) and `X` (10) to make 4 and 9. 
- `X` can be placed before `L` (50) and `C` (100) to make 40 and 90. 
- `C` can be placed before `D` (500) and `M` (1000) to make 400 and 900.

Given an integer, convert it to a roman numeral.

 

**Example 1:**

```
Input: num = 3
Output: "III"
```

**Example 2:**

```
Input: num = 4
Output: "IV"
```

**Example 3:**

```
Input: num = 9
Output: "IX"
```

```python
class Solution:
    def intToRoman(self, num: int) -> str:
        values = [1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1]
        numerals = ['M', 'CM', 'D', 'CD', 'C', 'XC', 'L', 'XL', 'X', 'IX', 'V', 'IV', 'I']
        result = ''
        for i in range(0, len(values)):
            while num >= values[i]:
                num -= values[i]
                result += numerals[i]
        return result
```



#### 13. Roman to Integer

Roman numerals are represented by seven different symbols: `I`, `V`, `X`, `L`, `C`, `D` and `M`.

```
Symbol       Value
I             1
V             5
X             10
L             50
C             100
D             500
M             1000
```

For example, `2` is written as `II` in Roman numeral, just two one's added together. `12` is written as `XII`, which is simply `X + II`. The number `27` is written as `XXVII`, which is `XX + V + II`.

Roman numerals are usually written largest to smallest from left to right. However, the numeral for four is not `IIII`. Instead, the number four is written as `IV`. Because the one is before the five we subtract it making four. The same principle applies to the number nine, which is written as `IX`. There are six instances where subtraction is used:

- `I` can be placed before `V` (5) and `X` (10) to make 4 and 9. 
- `X` can be placed before `L` (50) and `C` (100) to make 40 and 90. 
- `C` can be placed before `D` (500) and `M` (1000) to make 400 and 900.

Given a roman numeral, convert it to an integer.

 

**Example 1:**

```
Input: s = "III"
Output: 3
```

```python
class Solution:
    def romanToInt(self, s: str) -> int:
        res, prev = 0, 0
        dict = {'I':1, 'V':5, 'X':10, 'L':50, 'C':100, 'D':500, 'M':1000}
        for i in s[::-1]:          # rev the s
            if dict[i] >= prev:
                res += dict[i]     # sum the value iff previous value same or more
            else:
                res -= dict[i]     # substract when value is like "IV" --> 5-1, "IX" --> 10-1 etc 
            prev = dict[i]
        return res
```



#### 14. Longest Common Prefix

Write a function to find the longest common prefix string amongst an array of strings.

If there is no common prefix, return an empty string `""`.

 

**Example 1:**

```
Input: strs = ["flower","flow","flight"]
Output: "fl"
```

```python
class Solution:
    def longestCommonPrefix(self, strs: List[str]) -> str:
        if not strs:
            return ""
        
        for i in range(len(strs[0])):
            for string in strs[1:]:
                if i >= len(string) or string[i] != strs[0][i]:
                    return strs[0][:i]
        
        return strs[0]
```

```python
class Solution:
    def longestCommonPrefix(self, strs: List[str]) -> str:
        result = ""
        i = 0
        
        while True:
            try:
                sets = set(string[i] for string in strs)
                if len(sets) == 1:
                    result += sets.pop()
                    i += 1
                else:
                    break
            except Exception as e:
                break
                
        return result
```





#### 28. Implement strStr()

Implement [strStr()](http://www.cplusplus.com/reference/cstring/strstr/).

Return the index of the first occurrence of needle in haystack, or `-1` if `needle` is not part of `haystack`.

**Clarification:**

What should we return when `needle` is an empty string? This is a great question to ask during an interview.

For the purpose of this problem, we will return 0 when `needle` is an empty string. This is consistent to C's [strstr()](http://www.cplusplus.com/reference/cstring/strstr/) and Java's [indexOf()](https://docs.oracle.com/javase/7/docs/api/java/lang/String.html#indexOf(java.lang.String)).

 

**Example 1:**

```
Input: haystack = "hello", needle = "ll"
Output: 2
```

**Example 2:**

```
Input: haystack = "aaaaa", needle = "bba"
Output: -1
```

**Example 3:**

```
Input: haystack = "", needle = ""
Output: 0
```

```python
class Solution:
    def strStr(self, haystack: str, needle: str) -> int:
        for i in range(len(haystack) - len(needle) + 1):
                    if haystack[i:(i + len(needle))] == needle:
                        return i
                return -1
```

```python
class Solution:
    def strStr(self, haystack: str, needle: str) -> int:
        return haystack.find(needle)
```

```python
# KMP
class Solution:
    def strStr(self, haystack: str, needle: str) -> int:
        a = len(needle)
        b = len(haystack)
        if a == 0:
            return 0
        next = self.getnext(needle)
        p = 0
        for j in range(b):
            while p > 0 and haystack[j] != needle[p]:
                p = next[p-1]
            if haystack[j] == needle[p]:
                p += 1
            if p == a:
                return j - a + 1
        return -1

    def getnext(self, s):
        next = [0] * len(s)
        j = 0
        for i in range(1, len(s)):
            while (j > 0 and s[i] != s[j]):
                j = next[j-1]
            if s[i] == s[j]:
                j += 1
            next[i] = j
        return next
```



#### 38. Count and Say

The **count-and-say** sequence is a sequence of digit strings defined by the recursive formula:

- `countAndSay(1) = "1"`
- `countAndSay(n)` is the way you would "say" the digit string from `countAndSay(n-1)`, which is then converted into a different digit string.

To determine how you "say" a digit string, split it into the **minimal** number of groups so that each group is a contiguous section all of the **same character.** Then for each group, say the number of characters, then say the character. To convert the saying into a digit string, replace the counts with a number and concatenate every saying.

For example, the saying and conversion for digit string `"3322251"`:

![img](https://assets.leetcode.com/uploads/2020/10/23/countandsay.jpg)

Given a positive integer `n`, return *the* `nth` *term of the **count-and-say** sequence*.

 

**Example 1:**

```
Input: n = 1
Output: "1"
Explanation: This is the base case.
```

**Example 2:**

```
Input: n = 4
Output: "1211"
Explanation:
countAndSay(1) = "1"
countAndSay(2) = say "1" = one 1 = "11"
countAndSay(3) = say "11" = two 1's = "21"
countAndSay(4) = say "21" = one 2 + one 1 = "12" + "11" = "1211"
```

```python
class Solution:
    def countAndSay(self, n: int) -> str:
        s = '1'
        for _ in range(n-1):
            strnum, temp, count = s[0], '', 0
            for ch in s:
                if strnum == ch:
                    count += 1
                else:
                    temp += str(count) + strnum
                    strnum = ch
                    count = 1
            temp += str(count) + strnum
            s = temp
        return s
```





#### 49. Group Anagrams

Given an array of strings `strs`, group **the anagrams** together. You can return the answer in **any order**.

An **Anagram** is a word or phrase formed by rearranging the letters of a different word or phrase, typically using all the original letters exactly once.

 

**Example 1:**

```
Input: strs = ["eat","tea","tan","ate","nat","bat"]
Output: [["bat"],["nat","tan"],["ate","eat","tea"]]
```

```python
class Solution:
    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        solution = {}
        for i in range(len(strs)):
            reg = strs[i]
            regsort = ''.join(sorted(reg))
            if regsort in solution:
                solution[regsort].append(reg)
            else:
                solution[regsort] = [reg]
        return solution.values()
```



#### 67. Add Binary

Given two binary strings `a` and `b`, return *their sum as a binary string*.

 

**Example 1:**

```
Input: a = "11", b = "1"
Output: "100"
```

```python
class Solution:
    def addBinary(self, a: str, b: str) -> str:
        return(bin(int(a, base=2) + int(b, base=2))[2:]) 
```

```python
class Solution:
    def addBinary(self, a: str, b: str) -> str:
        result, carry, val = '', 0, 0
        for i in range(max(len(a), len(b))):
            val = carry
            if i < len(a):
                val += int(a[-i-1])
            if i < len(b):
                val += int(b[-i-1])
            carry, val = val // 2, val % 2
            result += str(val)
        if carry:
            result += str(1)
        return result[::-1]
```



#### 71. Simplify Path

Given a string `path`, which is an **absolute path** (starting with a slash `'/'`) to a file or directory in a Unix-style file system, convert it to the simplified **canonical path**.

**Example 1:**

```
Input: path = "/home/"
Output: "/home"
Explanation: Note that there is no trailing slash after the last directory name.
```

**Example 2:**

```
Input: path = "/../"
Output: "/"
Explanation: Going one level up from the root directory is a no-op, as the root level is the highest level you can go.
```

**Example 3:**

```
Input: path = "/home//foo/"
Output: "/home/foo"
Explanation: In the canonical path, multiple consecutive slashes are replaced by a single one.
```

**Example 4:**

```
Input: path = "/a/./b/../../c/"
Output: "/c"
```

```python
class Solution:
    def simplifyPath(self, path: str) -> str:
        result = []
        path_list = path.split('/')
        for p in path_list:
            if p:
                if p == '..':
                    if result:
                        result.pop()
                elif p == '.':
                    continue
                else:
                    result.append(p)
        res = '/' + '/'.join(result)
        return res
```



#### 125. Valid Palindrome

Given a string `s`, determine if it is a palindrome, considering only alphanumeric characters and ignoring cases.

 

**Example 1:**

```
Input: s = "A man, a plan, a canal: Panama"
Output: true
Explanation: "amanaplanacanalpanama" is a palindrome.
```

```python
class Solution:
    def isPalindrome(self, s: str) -> bool:
        left = 0
        right = len(s) - 1
        while left < right:
            while left < right and not s[left].isalnum():
                left += 1
            while left < right and not s[right].isalnum():
                right -= 1
            if s[left].lower() != s[right].lower():
                return False
            left += 1
            right -= 1
            
        return True
```



#### 150. Evaluate Reverse Polish Notation

Evaluate the value of an arithmetic expression in [Reverse Polish Notation](http://en.wikipedia.org/wiki/Reverse_Polish_notation).

Valid operators are `+`, `-`, `*`, and `/`. Each operand may be an integer or another expression.

```python
def evalRPN(tokens) -> int:
    stack = list()
    for i in range(len(tokens)):
        if tokens[i] not in ["+", "-", "*", "/"]:
            stack.append(tokens[i])
        else:
            tmp1 = stack.pop()
            tmp2 = stack.pop()
            res = eval(tmp2+tokens[i]+tmp1)
            stack.append(str(int(res)))
    return stack[-1]
```





#### 151. Reverse Words in a String

Given an input string `s`, reverse the order of the **words**.

A **word** is defined as a sequence of non-space characters. The **words** in `s` will be separated by at least one space.

Return *a string of the words in reverse order concatenated by a single space.*

**Note** that `s` may contain leading or trailing spaces or multiple spaces between two words. The returned string should only have a single space separating the words. Do not include any extra spaces.

 

**Example 1:**

```
Input: s = "the sky is blue"
Output: "blue is sky the"
```

```python
class Solution:
    def reverseWords(self, s: str) -> str:
        return " ".join(s.split()[::-1])
```



#### 168. Excel Sheet Column Title

Given an integer `columnNumber`, return *its corresponding column title as it appears in an Excel sheet*.

For example:

```
A -> 1
B -> 2
C -> 3
...
Z -> 26
AA -> 27
AB -> 28 
...
```

**Example 1:**

```
Input: columnNumber = 1
Output: "A"
```

```python
class Solution:
    def convertToTitle(self, n: int) -> str:
        result = ""
        while n:
            result += chr((n - 1) % 26 + ord("A"))
            n = (n - 1) //26
        return result[::-1]
        
```



#### 171. Excel Sheet Column Number

Given a string `columnTitle` that represents the column title as appear in an Excel sheet, return *its corresponding column number*.

For example:

```
A -> 1
B -> 2
C -> 3
...
Z -> 26
AA -> 27
AB -> 28 
...
```

**Example 1:**

```
Input: columnTitle = "A"
Output: 1
```

```python
class Solution:
    def titleToNumber(self, s: str) -> int:
        result = 0
        
        # Decimal 65 in ASCII corresponds to char 'A'
        alpha_map = {chr(i + 65): i + 1 for i in range(26)}

        n = len(s)
        for i in range(n):
            cur_char = s[n - 1 - i]
            result += (alpha_map[cur_char] * (26 ** i))
        return result
```

```python
class Solution:
    def titleToNumber(self, s: str) -> int:
        result = 0
        n = len(s)
        for i in range(n):
            result = result * 26
            result += (ord(s[i]) - ord('A') + 1)
        return result
```

#### 205. Isomorphic Strings

Given two strings `s` and `t`, *determine if they are isomorphic*.

Two strings `s` and `t` are isomorphic if the characters in `s` can be replaced to get `t`.

All occurrences of a character must be replaced with another character while preserving the order of characters. No two characters may map to the same character, but a character may map to itself.

 

**Example 1:**

```
Input: s = "egg", t = "add"
Output: true
```

```python
class Solution:
    def isIsomorphic(self, s: str, t: str) -> bool:
        return len(set(s)) == len(set(t)) and len(set(s)) == len(set(zip(s, t)))
```

#### 228. Summary Ranges

You are given a **sorted unique** integer array `nums`.

Return *the **smallest sorted** list of ranges that **cover all the numbers in the array exactly***. That is, each element of `nums` is covered by exactly one of the ranges, and there is no integer `x` such that `x` is in one of the ranges but not in `nums`.

Each range `[a,b]` in the list should be output as:

- `"a->b"` if `a != b`
- `"a"` if `a == b`

 

**Example 1:**

```
Input: nums = [0,1,2,4,5,7]
Output: ["0->2","4->5","7"]
Explanation: The ranges are:
[0,2] --> "0->2"
[4,5] --> "4->5"
[7,7] --> "7"
```

```python
class Solution:
    def summaryRanges(self, nums: List[int]) -> List[str]:
        n = len(nums)
        if n == 0:
            return []
        if n == 1:
            return [str(nums[0])]
        output = []
        start = nums[0]
        for i in range(1, n):
            if nums[i] - nums[i-1] > 1:
                end = nums[i-1]
                if end == start:
                    output.append(str(start))
                else:
                    output.append(str(start) + "->" + str(end))
                start = nums[i]
            if i == n - 1:
                if nums[i] - nums[i-1] > 1:
                    output.append(str(start))
                else:
                    output.append(str(start) + "->" + str(nums[i]))
        return output
```



### List

#### 19. Remove Nth Node From End of List

Given the `head` of a linked list, remove the `nth` node from the end of the list and return its head.

**Follow up:** Could you do this in one pass?

 

**Example 1:**

![img](https://assets.leetcode.com/uploads/2020/10/03/remove_ex1.jpg)

```
Input: head = [1,2,3,4,5], n = 2
Output: [1,2,3,5]
```

**Example 2:**

```
Input: head = [1], n = 1
Output: []
```

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def removeNthFromEnd(self, head, n):
        def remove(head):
            if not head:
                return 0, head
            i, head.next = remove(head.next)
            return i + 1, (head, head.next)[i + 1 == n]
            
        return remove(head)[1]
```

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def removeNthFromEnd(self, head, n):
        slow = fast = head
        for _ in range(n):
            fast = fast.next
        if not fast:
            return head.next
        while fast.next:
            slow = slow.next
            fast = fast.next
        slow.next = slow.next.next
        return head
```



#### 21. Merge Two Sorted Lists

Merge two sorted linked lists and return it as a **sorted** list. The list should be made by splicing together the nodes of the first two lists.

 

**Example 1:**

![img](https://assets.leetcode.com/uploads/2020/10/03/merge_ex1.jpg)

```
Input: l1 = [1,2,4], l2 = [1,3,4]
Output: [1,1,2,3,4,4]
```

```python
class Solution:
    def mergeTwoLists(self, l1: ListNode, l2: ListNode) -> ListNode:
        curr = dummy = ListNode(0)
        
        while l1 and l2:
            if l1.val < l2.val:
                curr.next = l1
                l1 = l1.next
            else:
                curr.next = l2
                l2 = l2.next
            curr = curr.next
        curr.next = l1 or l2
        
        return dummy.next
```

```python
class Solution:
    def mergeTwoLists(self, l1: ListNode, l2: ListNode) -> ListNode:
        if not l1 or not l2:
            return l1 or l2
        if l1.val < l2.val:
            l1.next = self.mergeTwoLists(l1.next, l2)
            return l1
        else:
            l2.next = self.mergeTwoLists(l1, l2.next)
            return l2
```



#### 143*. Reorder List

You are given the head of a singly linked-list. The list can be represented as:

```
L0 → L1 → … → Ln - 1 → Ln
```

*Reorder the list to be on the following form:*

```
L0 → Ln → L1 → Ln - 1 → L2 → Ln - 2 → …
```

You may not modify the values in the list's nodes. Only nodes themselves may be changed.

 

**Example 1:**

![img](https://assets.leetcode.com/uploads/2021/03/04/reorder1linked-list.jpg)

```
Input: head = [1,2,3,4]
Output: [1,4,2,3]
```

**Example 2:**

![img](https://assets.leetcode.com/uploads/2021/03/09/reorder2-linked-list.jpg)

```
Input: head = [1,2,3,4,5]
Output: [1,5,2,4,3]
```

```python
# 线性表
class Solution:
    def reorderList(self, head: ListNode) -> None:
        """
        Do not return anything, modify head in-place instead.
        """
        if not head:
            return
        vec = []
        node = head
        while node:
            vec.append(node)
            node = node.next
            
        i, j = 0, len(vec) - 1
        while i < j:
            vec[i].next = vec[j]
            i += 1
            if i == j:
                break
            vec[j].next = vec[i]
            j -= 1
            
        vec[i].next = None
```

```python
# 寻找链表中点 + 链表逆序 + 合并链表
class Solution:
    def reorderList(self, head: ListNode) -> None:
        if not head:
            return 
        mid = self.middleNode(head)
        l1 = head
        l2 = mid.next
        mid.next = None
        l2 = self.reverseList(l2)
        self.mergeList(l1, l2)
        
    def middleNode(self, head):
        slow = fast = head
        while fast.next and fast.next.next:
        	slow = slow.next
        	fast = fast.next.next
        return slow
    
    def reverList(self, head):
        prev = None
        curr = head
        while curr:
            temp = head.next
            head.next = prev
            prev = head
            head = temp
        return prev
    
    def mergeList(self, l1, l2):
        while l1 and l2:
            l1_tmp = l1.next
            l2_tmp = l2.next
            
            l1.next = l2
            l1 = l1_tmp
            
            l2.next = l1
            l2 = l2_tmp
```



#### 147. Insertion Sort List

Given the `head` of a singly linked list, sort the list using **insertion sort**, and return *the sorted list's head*.

The steps of the **insertion sort** algorithm:

1. Insertion sort iterates, consuming one input element each repetition and growing a sorted output list.
2. At each iteration, insertion sort removes one element from the input data, finds the location it belongs within the sorted list and inserts it there.
3. It repeats until no input elements remain.

The following is a graphical example of the insertion sort algorithm. The partially sorted list (black) initially contains only the first element in the list. One element (red) is removed from the input data and inserted in-place into the sorted list with each iteration.

![img](https://upload.wikimedia.org/wikipedia/commons/0/0f/Insertion-sort-example-300px.gif)

 

**Example 1:**

![img](https://assets.leetcode.com/uploads/2021/03/04/sort1linked-list.jpg)

```
Input: head = [4,2,1,3]
Output: [1,2,3,4]
```

```python
class Solution:
    def insertionSortList(self, head: ListNode) -> ListNode:
        if not head:
            return head
        
        dummyHead = ListNode(0, head)
        lastSorted = head
        curr = head.next

        while curr:
            if lastSorted.val <= curr.val: # curr不需要插入
                lastSorted = lastSorted.next
            else: # 需要插入curr到前面的某一位置
                prev = dummyHead
                while prev.next.val <= curr.val: # 找到应该插入的位置
                    prev = prev.next
                lastSorted.next = curr.next # 末位链接curr下一位（提取curr）
                curr.next = prev.next # 插入curr
                prev.next = curr
            curr = lastSorted.next
        
        return dummyHead.next
```



#### 2. Add Two Numbers

You are given two **non-empty** linked lists representing two non-negative integers. The digits are stored in **reverse order**, and each of their nodes contains a single digit. Add the two numbers and return the sum as a linked list.

You may assume the two numbers do not contain any leading zero, except the number 0 itself.

 

**Example 1:**

![img](https://assets.leetcode.com/uploads/2020/10/02/addtwonumber1.jpg)

```
Input: l1 = [2,4,3], l2 = [5,6,4]
Output: [7,0,8]
Explanation: 342 + 465 = 807.
```

```python
class Solution:
    def addTwoNumbers(self, l1: ListNode, l2: ListNode) -> ListNode:
        dummy = tail = ListNode(0)
        s = 0
        while l1 or l2 or s: # or s 保证了最后的进位
            s += (l1.val if l1 else 0) + (l2.val if l2 else 0)
            tail.next = ListNode(s % 10)
            tail = tail.next
            s //= 10
            l1 = l1.next if l1 else None
            l2 = l2.next if l2 else None
        return dummy.next
```



#### 82. Remove Duplicates from Sorted List II

Given the `head` of a sorted linked list, *delete all nodes that have duplicate numbers, leaving only distinct numbers from the original list*. Return *the linked list **sorted** as well*.

 

**Example 1:**

![img](https://assets.leetcode.com/uploads/2021/01/04/linkedlist1.jpg)

```
Input: head = [1,2,3,3,4,4,5]
Output: [1,2,5]
```

**Example 2:**

![img](https://assets.leetcode.com/uploads/2021/01/04/linkedlist2.jpg)

```
Input: head = [1,1,1,2,3]
Output: [2,3]
```

```python
class Solution:
    def deleteDuplicates(self, head: ListNode) -> ListNode:
        if not head or not head.next:
            return head
                    
        prev = ListNode(0, head)
        
        curr = prev
        while curr.next and curr.next.next:
            if curr.next.val == curr.next.next.val:
                x = curr.next.val
                while curr.next and curr.next.val == x:
                    curr.next = curr.next.next # curr does not change
            else:
                curr = curr.next
        
        return prev.next
```



#### 83. Remove Duplicates from Sorted List

Given the `head` of a sorted linked list, *delete all duplicates such that each element appears only once*. Return *the linked list **sorted** as well*.

 

**Example 1:**

![img](https://assets.leetcode.com/uploads/2021/01/04/list1.jpg)

```
Input: head = [1,1,2]
Output: [1,2]
```

**Example 2:**

![img](https://assets.leetcode.com/uploads/2021/01/04/list2.jpg)

```
Input: head = [1,1,2,3,3]
Output: [1,2,3]
```

```python
class Solution:
    def deleteDuplicates(self, head: ListNode) -> ListNode:
        if not head:
            return head
        curr = head.next
        prev = head
        while curr:
            if curr.val == prev.val:
                prev.next = curr.next
            else:
                prev = curr
            curr = curr.next
        return head
```

```python
class Solution:
    def deleteDuplicates(self, head: ListNode) -> ListNode:
        if not head or not head.next:
            return head
        curr = head
        while curr and curr.next:
            if curr.next.val == curr.val:
                x = curr.val
                while curr.next and curr.next.val == x:
                    curr.next = curr.next.next # curr does not change
            else:
                curr = curr.next
        return head
```



#### 86. Partition List

Given the `head` of a linked list and a value `x`, partition it such that all nodes **less than** `x` come before nodes **greater than or equal** to `x`.

You should **preserve** the original relative order of the nodes in each of the two partitions.

 

**Example 1:**

![img](https://assets.leetcode.com/uploads/2021/01/04/partition.jpg)

```
Input: head = [1,4,3,2,5,2], x = 3
Output: [1,2,2,4,3,5]
```

**Example 2:**

```
Input: head = [2,1], x = 2
Output: [1,2]
```

```python
class Solution:
    def partition(self, head: ListNode, x: int) -> ListNode:
        # 1. two list
        # 2. one pass
        # 3. merge
        
        head1 = ListNode(0)
        head2 = ListNode(0)
        
        curr = head
        l1, l2 = head1, head2
        while curr:
            if curr.val < x:
                l1.next = curr
                l1 = curr
            else:
                l2.next = curr
                l2 = curr
            
            curr = curr.next
        l2.next = None
        l1.next = head2.next
        
        return head1.next
```





#### 24. Swap Nodes in Pairs

Given a linked list, swap every two adjacent nodes and return its head. You must solve the problem without modifying the values in the list's nodes (i.e., only nodes themselves may be changed.)

 

**Example 1:**

![img](https://assets.leetcode.com/uploads/2020/10/03/swap_ex1.jpg)

```
Input: head = [1,2,3,4]
Output: [2,1,4,3]
```

**Example 2:**

```
Input: head = []
Output: []
```

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def swapPairs(self, head: ListNode) -> ListNode:
        dummyHead = ListNode(0, head)
        temp = dummyHead
        while temp.next and temp.next.next:
            node1 = temp.next
            node2 = temp.next.next
            temp.next = node2          # step 1
            node1.next = node2.next    # step 2
            node2.next = node1         # step 3
            temp = node1               # update starting point
        return dummyHead.next
```

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def swapPairs(self, head: ListNode) -> ListNode:
        if not head or not head.next:
            return head
        newHead = head.next
        head.next = self.swapPairs(newHead.next)
        newHead.next = head
        return newHead
```

#### 92. Reverse Linked List II

Given the `head` of a singly linked list and two integers `left` and `right` where `left <= right`, reverse the nodes of the list from position `left` to position `right`, and return *the reversed list*.

 **Note**: re-connect after reverse.

**Example 1:**

![img](https://assets.leetcode.com/uploads/2021/02/19/rev2ex2.jpg)

```
Input: head = [1,2,3,4,5], left = 2, right = 4
Output: [1,4,3,2,5]
```

**Example 2:**

```
Input: head = [5], left = 1, right = 1
Output: [5]
```

```python
class Solution:
    def reverseBetween(self, head: ListNode, left: int, right: int) -> ListNode:
        dummyNode = ListNode(-1, head)
        m_prev = self.findkth(dummyNode, left-1)
        m = m_prev.next
        n = self.findkth(m, right-left)
        n_next = n.next
        n.next = None
        
        self.reverse(m)
        
        m_prev.next = n
        m.next = n_next
        
        return dummyNode.next
    
    def reverse(self, head):
        prev = None
        while head:
            temp = head.next
            head.next = prev
            prev = head
            head = temp
        return prev
    
    def findkth(self, head, k):
        for i in range(k):
            if head is None:
                return None
            head = head.next
        return head
```



#### 61. Rotate List

Given the `head` of a linked list, rotate the list to the right by `k` places.

 

**Example 1:**

![img](https://assets.leetcode.com/uploads/2020/11/13/rotate1.jpg)

```
Input: head = [1,2,3,4,5], k = 2
Output: [4,5,1,2,3]
```

```python
class Solution:
    def rotateRight(self, head: ListNode, k: int) -> ListNode:
        if not head or not head.next:
            return head
        n = 0
        p = head
        while p:
            n += 1
            p = p.next
        k = k % n
        
        if k == 0:
            return head
        
        p1, p2 = self.kthFromEnd(head, k)
        newHead = p1.next
        
        p1.next = None
        p2.next = head
        
        return newHead
        
    def kthFromEnd(self, head, k):
        slow = fast = head
        for _ in range(k):
            fast = fast.next
        if not fast:
            return head.next
        while fast.next:
            slow = slow.next
            fast = fast.next
        return slow, fast
```

```python
# 方法2： 闭合为环做剪切
class Solution:
    def rotateRight(self, head: ListNode, k: int) -> ListNode:
        if k == 0 or not head or not head.next:
            return head
        
        n = 1
        cur = head
        while cur.next:             # 计算List长度并把curr指向末尾
            cur = cur.next
            n += 1
        
        if (add := n - k % n) == n: # 如果k是n的倍数，则什么都不必做
            return head
        
        cur.next = head             # 闭合成环
        while add:                  # 将curr移到add处
            cur = cur.next
            add -= 1
        
        head = cur.next             # 在curr处剪断闭环
        cur.next = None
        return head
```









#### 141. Linked List Cycle

Given `head`, the head of a linked list, determine if the linked list has a cycle in it.

There is a cycle in a linked list if there is some node in the list that can be reached again by continuously following the `next` pointer. Internally, `pos` is used to denote the index of the node that tail's `next` pointer is connected to. **Note that `pos` is not passed as a parameter**.

Return `true` *if there is a cycle in the linked list*. Otherwise, return `false`.

 

**Example 1:**

![img](https://assets.leetcode.com/uploads/2018/12/07/circularlinkedlist.png)

```
Input: head = [3,2,0,-4], pos = 1
Output: true
Explanation: There is a cycle in the linked list, where the tail connects to the 1st node (0-indexed).
```

```python
# 方法一：哈希表
class Solution:
    def hasCycle(self, head: ListNode) -> bool:
        seen = set()
        while head:
            if head in seen:
                return True
            seen.add(head)
            head = head.next
        return False
```

```python
# 方法二：快慢指针
class Solution:
    def hasCycle(self, head: ListNode) -> bool:
        fast = slow = head
        while fast and fast.next:
            slow, fast = slow.next, fast.next.next
            if slow == fast:
                return True
        return False
        
```



#### 142. Linked List Cycle II

Given a linked list, return the node where the cycle begins. If there is no cycle, return `null`.

There is a cycle in a linked list if there is some node in the list that can be reached again by continuously following the `next` pointer. Internally, `pos` is used to denote the index of the node that tail's `next` pointer is connected to. **Note that `pos` is not passed as a parameter**.

**Notice** that you **should not modify** the linked list.

 

**Example 1:**

![img](https://assets.leetcode.com/uploads/2018/12/07/circularlinkedlist.png)

```
Input: head = [3,2,0,-4], pos = 1
Output: tail connects to node index 1
Explanation: There is a cycle in the linked list, where tail connects to the second node.
```

```python
class Solution:
    def detectCycle(self, head: ListNode) -> ListNode:
        seen = set()
        while head:
            if head not in seen:
                seen.add(head)
            else:
                return head
            head = head.next
        return None
```

```python
class Solution:
    def detectCycle(self, head: ListNode) -> ListNode:
        slow = fast = head
        while fast and fast.next:
            slow, fast = slow.next, fast.next.next
            if slow == fast:
                curr = head
                while curr != slow:
                    curr = curr.next
                    slow = slow.next
                return curr
        return None 
```





#### 148*. Sort List

Given the `head` of a linked list, return *the list after sorting it in **ascending order***.

**Follow up:** Can you sort the linked list in `O(n logn)` time and `O(1)` memory (i.e. constant space)?

 

**Example 1:**

![img](https://assets.leetcode.com/uploads/2020/09/14/sort_list_1.jpg)

```
Input: head = [4,2,1,3]
Output: [1,2,3,4]
```

```python 
#归并排序
class Solution:
    def sortList(self, head):
        if not head or not head.next:
            return head
        slow, fast = head, head.next
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
        head2 = slow.next
        slow.next = None # 断开
        return self.merge(self.sortList(head), self.sortList(head2))
    
    def merge(self, head1, head2):
        dummyhead = curr = ListNode(0)
        while head1 and head2:
            if head1.val < head2.val:
                curr.next = head1
                head1 = head1.next
            else:
                curr.next = head2
                head2 = head2.next
            curr = curr.next
        curr.next = head1 or head2
        return dummyhead.next
```





#### 160. Intersection of Two Linked Lists

**comment**: trick problem

Given the heads of two singly linked-lists `headA` and `headB`, return *the node at which the two lists intersect*. If the two linked lists have no intersection at all, return `null`.

For example, the following two linked lists begin to intersect at node `c1`:

![img](https://assets.leetcode.com/uploads/2021/03/05/160_statement.png)

It is **guaranteed** that there are no cycles anywhere in the entire linked structure.

**Note** that the linked lists must **retain their original structure** after the function returns.

```python
class Solution:
    def getIntersectionNode(self, headA: ListNode, headB: ListNode) -> ListNode:
        p1 = headA
        p2 = headB
        
        while p1 != p2:
            p1 = p1.next if p1 else headB
            p2 = p2.next if p2 else headA
        
        return p1
```



#### 234. Palindrome Linked List

```python

class Solution:
    def isPalindrome(self, head: ListNode) -> bool:

        self.front_pointer = head

        def recursively_check(current_node=head):
            if current_node is not None:
                if not recursively_check(current_node.next):
                    return False
                if self.front_pointer.val != current_node.val:
                    return False
                self.front_pointer = self.front_pointer.next
            return True

        return recursively_check()
```



### Tree

#### 144. Binary Tree Preorder Traversal

Given the `root` of a binary tree, return *the preorder traversal of its nodes' values*.

```python
class Solution:
    def preorderTraversal(self, root: TreeNode) -> List[int]:
        # 保存结果
        result = []
        
        def traversal(root: TreeNode):
            if root == None:
                return
            result.append(root.val) # 前序
            traversal(root.left)    # 左
            traversal(root.right)   # 右

        traversal(root)
        return result
```

```python
class Solution:
    def preorderTraversal(self, root: TreeNode) -> List[int]:
        # 根结点为空则返回空列表
        if not root:
            return []
        stack = [root]
        result = []
        while stack:
            node = stack.pop()
            result.append(node.val) # 中结点先处理
            if node.right:
                stack.append(node.right) # 右孩子先入栈
            if node.left:
                stack.append(node.left) # 左孩子后入栈
        return result
```

```python
class Solution:
    def preorderTraversal(self, root: TreeNode) -> List[int]:
        result = []
        st= []
        if root:
            st.append(root)
        while st:
            node = st.pop()
            if node != None:
                if node.right: #右
                    st.append(node.right)
                if node.left: #左
                    st.append(node.left)
                st.append(node) #中
                st.append(None)
            else:
                node = st.pop()
                result.append(node.val)
        return result
```



#### 145. Binary Tree Postorder Traversal

Given the `root` of a binary tree, return *the postorder traversal of its nodes' values*.

```python
class Solution:
    def postorderTraversal(self, root: TreeNode) -> List[int]:
        result = []

        def traversal(root: TreeNode):
            if root == None:
                return
            traversal(root.left)    # 左
            traversal(root.right)   # 右
            result.append(root.val) # 中

        traversal(root)
        return result
```

```python
class Solution:
    def postorderTraversal(self, root: TreeNode) -> List[int]:
        if not root:
            return []
        stack = [root]
        result = []
        while stack:
            node = stack.pop()
            # 中结点先处理
            result.append(node.val)
            # 左孩子先入栈
            if node.left:
                stack.append(node.left)
            # 右孩子后入栈
            if node.right:
                stack.append(node.right)
        # 将最终的数组翻转
        return result[::-1]
```

```python
class Solution:
    def postorderTraversal(self, root: TreeNode) -> List[int]:
        result = []
        st = []
        if root:
            st.append(root)
        while st:
            node = st.pop()
            if node != None:
                st.append(node) #中
                st.append(None)
                
                if node.right: #右
                    st.append(node.right)
                if node.left: #左
                    st.append(node.left)
            else:
                node = st.pop()
                result.append(node.val)
        return result
```





#### 94. Binary Tree Inorder Traversal

Given the `root` of a binary tree, return *the inorder traversal of its nodes' values*.

```python
class Solution:
    def inorderTraversal(self, root: TreeNode) -> List[int]:
        result = []

        def traversal(root: TreeNode):
            if root == None:
                return
            traversal(root.left)    # 左
            result.append(root.val) # 中
            traversal(root.right)   # 右

        traversal(root)
        return result
```

```python
class Solution:
    def inorderTraversal(self, root: TreeNode) -> List[int]:
        if not root:
            return []
        stack = []  # 不能提前将root结点加入stack中
        result = []
        cur = root
        while cur or stack:
            # 先迭代访问最底层的左子树结点
            if cur:     
                stack.append(cur)
                cur = cur.left		
            # 到达最左结点后处理栈顶结点    
            else:		
                cur = stack.pop()
                result.append(cur.val)
                # 取栈顶元素右结点
                cur = cur.right	
        return result
```

```python
class Solution:
    def inorderTraversal(self, root: TreeNode) -> List[int]:
        result = []
        st = []
        if root:
            st.append(root)
        while st:
            node = st.pop()
            if node != None:
                if node.right: #添加右节点（空节点不入栈）
                    st.append(node.right)
                
                st.append(node) #添加中节点
                st.append(None) #中节点访问过，但是还没有处理，加入空节点做为标记。
                
                if node.left: #添加左节点（空节点不入栈）
                    st.append(node.left)
            else: #只有遇到空节点的时候，才将下一个节点放进结果集
                node = st.pop() #重新取出栈中元素
                result.append(node.val) #加入到结果集
        return result
```

#### 102. Binary Tree Level Order Traversal

Given the `root` of a binary tree, return *the level order traversal of its nodes' values*. (i.e., from left to right, level by level).

```python
class Solution:
    def levelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
        if not root:
            return []
        queue = [root]
        result = []
        while queue:
            length = len(queue)  
            layer = []
            for _ in range(length):
                curr = queue.pop(0)
                layer.append(curr.val)
                if curr.left: queue.append(curr.left)
                if curr.right: queue.append(curr.right)
            result.append(layer)
        return result
```

#### 107. Binary Tree Level Order Traversal II

#### 199. Binary Tree Right Side View

#### 637. Average of Levels in Binary Tree

#### 429. N-ary Tree Level Order Traversal

#### 515. Find Largest Value in Each Tree Row

#### 116. Populating Next Right Pointers in Each Node

#### 117. Populating Next Right Pointers in Each Node II

#### 104. Maximum Depth of Binary Tree

```python
def maxDepth(self, root: TreeNode) -> int:
    if root is None:
        return 0 
    return max(self.maxDepth(root.left), self.maxDepth(root.right)) + 1
```

```python
class solution:
    def maxdepth(self, root: treenode) -> int:
        if not root:
            return 0
        depth = 0 #记录深度
        queue = collections.deque()
        queue.append(root)
        while queue:
            size = len(queue)
            depth += 1
            for i in range(size):
                node = queue.popleft()
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
        return depth
```

#### 559. Maximum Depth of N-ary Tree

#### 111. Minimum Depth of Binary Tree

只有当左右孩子都为空的时候，才说明遍历的最低点了。如果其中一个孩子为空则不是最低点

```python
def minDepth(self, root: TreeNode) -> int:
    if root is None:
        return 0
    if root.left and root.right:
        return min(self.minDepth(root.left), self.minDepth(root.right)) + 1
    else:
        return max(self.minDepth(root.left), self.minDepth(root.right)) + 1
```

```python
class Solution:
    def minDepth(self, root: TreeNode) -> int:
        if not root:
            return 0
        que = deque()
        que.append(root)
        res = 1

        while que:
            for _ in range(len(que)):
                node = que.popleft()
                # 当左右孩子都为空的时候，说明是最低点的一层了，退出
                if not node.left and not node.right:
                    return res
                if node.left is not None:
                    que.append(node.left)
                if node.right is not None:
                    que.append(node.right)
            res += 1
        return res
```



#### 110. Balanced Binary Tree

递归法

```python
def isBalanced(self, root: TreeNode) -> bool:
    def get_height(root):
        if root is None:
            return 0
        left_height, right_height = get_height(root.left), get_height(root.right)
        if left_height < 0 or right_height < 0 or abs(left_height - right_height) > 1:
            return -1
        return max(left_height, right_height) + 1
    return get_height(root) >= 0
```

```python
class Solution:
    def isBalanced(self, root: TreeNode) -> bool:
        return True if self.getDepth(root) != -1 else False
    
    #返回以该节点为根节点的二叉树的高度，如果不是二叉搜索树了则返回-1
    def getDepth(self, node):
        if not node:
            return 0
        leftDepth = self.getDepth(node.left)
        if leftDepth == -1: return -1 #说明左子树已经不是二叉平衡树
        rightDepth = self.getDepth(node.right)
        if rightDepth == -1: return -1 #说明右子树已经不是二叉平衡树
        return -1 if abs(leftDepth - rightDepth)>1 else 1 + max(leftDepth, rightDepth)
```

迭代法

```python
class Solution:
    def isBalanced(self, root: TreeNode) -> bool:
        st = []
        if not root:
            return True
        st.append(root)
        while st:
            node = st.pop() #中
            if abs(self.getDepth(node.left) - self.getDepth(node.right)) > 1:
                return False
            if node.right:
                st.append(node.right) #右（空节点不入栈）
            if node.left:
                st.append(node.left) #左（空节点不入栈）
        return True
    
    def getDepth(self, cur):
        st = []
        if cur:
            st.append(cur)
        depth = 0
        result = 0
        while st:
            node = st.pop()
            if node:
                st.append(node) #中
                st.append(None)
                depth += 1
                if node.right: st.append(node.right) #右
                if node.left: st.append(node.left) #左
            else:
                node = st.pop()
                depth -= 1
            result = max(result, depth)
        return result
```



#### 589. N-ary Tree Preorder Traversal

Given the `root` of an n-ary tree, return *the preorder traversal of its nodes' values*.

Nary-Tree input serialization is represented in their level order traversal. Each group of children is separated by the null value (See examples)

 

**Example 1:**

![img](https://assets.leetcode.com/uploads/2018/10/12/narytreeexample.png)

```
Input: root = [1,null,3,2,4,null,5,6]
Output: [1,3,5,6,2,4]
```

```python
class Solution:
    def preorder(self, root: 'Node') -> List[int]:
        result = []
        def traversal(roots):
            if not roots:
                return
            for node in roots:
                if node:
                    result.append(node.val)
                    traversal(node.children)
        traversal([root])
        return result
```

#### 226. Invert Binary Tree

递归法：前序遍历：

```python
class Solution:
    def invertTree(self, root: TreeNode) -> TreeNode:
        if not root:
            return None
        root.left, root.right = root.right, root.left #中
        self.invertTree(root.left) #左
        self.invertTree(root.right) #右
        return root
```

迭代法：深度优先遍历（前序遍历）：

```python
class Solution:
    def invertTree(self, root: TreeNode) -> TreeNode:
        if not root:
            return root
        st = []
        st.append(root)
        while st:
            node = st.pop()
            node.left, node.right = node.right, node.left #中
            if node.right:
                st.append(node.right) #右
            if node.left:
                st.append(node.left) #左
        return root
```

迭代法：广度优先遍历（层序遍历）：

```python
import collections
class Solution:
    def invertTree(self, root: TreeNode) -> TreeNode:
        queue = collections.deque() #使用deque()
        if root:
            queue.append(root)
        while queue:
            size = len(queue)
            for i in range(size):
                node = queue.popleft()
                node.left, node.right = node.right, node.left #节点处理
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
        return root
```

#### 101. Symmetric Tree

递归法

```python
class Solution:
    def isSymmetric(self, root: TreeNode) -> bool:
        if not root:
            return True
        return self.compare(root.left, root.right)
        
    def compare(self, left, right):
        #首先排除空节点的情况
        if left == None and right != None: return False
        elif left != None and right == None: return False
        elif left == None and right == None: return True
        #排除了空节点，再排除数值不相同的情况
        elif left.val != right.val: return False
        
        #此时就是：左右节点都不为空，且数值相同的情况
        #此时才做递归，做下一层的判断
        outside = self.compare(left.left, right.right) #左子树：左、 右子树：右
        inside = self.compare(left.right, right.left) #左子树：右、 右子树：左
        isSame = outside and inside #左子树：中、 右子树：中 （逻辑处理）
        return isSame
```

迭代法： 使用队列

```python
import collections
class Solution:
    def isSymmetric(self, root: TreeNode) -> bool:
        if not root:
            return True
        queue = collections.deque()
        queue.append(root.left) #将左子树头结点加入队列
        queue.append(root.right) #将右子树头结点加入队列
        while queue: #接下来就要判断这这两个树是否相互翻转
            leftNode = queue.popleft()
            rightNode = queue.popleft()
            if not leftNode and not rightNode: #左节点为空、右节点为空，此时说明是对称的
                continue
            
            #左右一个节点不为空，或者都不为空但数值不相同，返回false
            if not leftNode or not rightNode or leftNode.val != rightNode.val:
                return False
            queue.append(leftNode.left) #加入左节点左孩子
            queue.append(rightNode.right) #加入右节点右孩子
            queue.append(leftNode.right) #加入左节点右孩子
            queue.append(rightNode.left) #加入右节点左孩子
        return True
```

迭代法：使用栈

```python
class Solution:
    def isSymmetric(self, root: TreeNode) -> bool:
        if not root:
            return True
        st = [] #这里改成了栈
        st.append(root.left)
        st.append(root.right)
        while st:
            leftNode = st.pop()
            rightNode = st.pop()
            if not leftNode and not rightNode:
                continue
            if not leftNode or not rightNode or leftNode.val != rightNode.val:
                return False
            st.append(leftNode.left)
            st.append(rightNode.right)
            st.append(leftNode.right)
            st.append(rightNode.left)
        return True
```

#### 100. Same Tree

```python
def isSameTree(self, p: TreeNode, q: TreeNode) -> bool:
    if p is None and q is None:
        return True
    if p is not None and q is not None:
        return p.val == q.val and self.isSameTree(p.left, q.left) and self.isSameTree(p.right, q.right)
    return False
```

#### 572. Subtree of Another Tree

```python
class Solution:
    def isSubtree(self, s: TreeNode, t: TreeNode) -> bool:
        return str(t) in str(s)
```

```python
class Solution(object):
    def isSubtree(self, s, t):
        def ser(root):
            if not root:    return '#'
            st = str(root.val) + str(ser(root.left)) + str(ser(root.right)) 
            return ' ' + st + ' ' #前后加空格，避免[12][1]的情况误判
        return ser(t) in ser(s)
```

```python
class Solution:
    def isSubtree(self, s: TreeNode, t: TreeNode) -> bool:
        def dfs_sequence(root, li):
            """
            返回 DFS 序列
            """
            if root is None:
                return li
            li.append('{' + str(root.val) + '}')
            if root.left is None:
                li.append('LeftNone')
            else:
                dfs_sequence(root.left, li)
            if root.right is None:
                li.append('RightNone')
            else:
                dfs_sequence(root.right, li)
            return li
        
        s = ','.join(dfs_sequence(s, []))
        t = ','.join(dfs_sequence(t, []))
        return t in s
```

```python
class Solution:
    def isSubtree(self, s: Optional[TreeNode], t: Optional[TreeNode]) -> bool:
        if not s and not t:
            return True
        if not s or not t:
            return False
        return self.isSameTree(s, t) or self.isSubtree(s.left, t) or self.isSubtree(s.right, t)

    
    def isSameTree(self, p, q):
        if p is None and q is None:
            return True
        if p is not None and q is not None:
            return p.val == q.val and self.isSameTree(p.left, q.left) and self.isSameTree(p.right, q.right)
        return False
```

#### 222. Count Complete Tree Nodes

```python
class Solution:
    def countNodes(self, root: Optional[TreeNode]) -> int:
        if not root:
            return 0
        return 1 + self.countNodes(root.left) + self.countNodes(root.right)
```

#### 257. Binary Tree Paths

回溯法

```python
class Solution:
    def binaryTreePaths(self, root: TreeNode) -> List[str]:
        if not root:
            return
        
        path=[root.val]
        res=[]
        def backtrace(root, path):
            if not root:return 
            if (not root.left) and (not root.right):
               res.append(path[:])
  
            for way in [root.left, root.right]:
                if not way: continue
                path.append(way.val)
                backtrace(way, path)
                path.pop()
        backtrace(root,path)
        return ["->".join(list(map(str,i))) for i in res]
```









#### 501. Find Mode in Binary Search Tree

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
# 递归法
class Solution:
    def findMode(self, root: TreeNode) -> List[int]:
        if not root: return
        self.pre = root
        self.count = 0   //统计频率
        self.countMax = 0  //最大频率
        self.res = []
        def findNumber(root):
            if not root: return None  // 第一个节点
            findNumber(root.left)  //左
            if self.pre.val == root.val:  //中: 与前一个节点数值相同
                self.count += 1
            else:  // 与前一个节点数值不同
                self.pre = root
                self.count = 1
            if self.count > self.countMax:  // 如果计数大于最大值频率
                self.countMax = self.count  // 更新最大频率
                self.res = [root.val]  //更新res
            elif self.count == self.countMax:  // 如果和最大值相同，放进res中
                self.res.append(root.val)
            findNumber(root.right)  //右
            return
        findNumber(root)
        return self.res
	

# 迭代法-中序遍历-使用额外空间map的方法：
class Solution:
    def findMode(self, root: TreeNode) -> List[int]:
        stack = []
        cur = root
        pre = None
        dist = {}
        while cur or stack:
            if cur:  # 指针来访问节点，访问到最底层
                stack.append(cur)
                cur = cur.left
            else:  # 逐一处理节点
                cur = stack.pop()
                if cur.val in dist:
                    dist[cur.val] += 1
                else:
                    dist[cur.val] = 1
                pre = cur
                cur = cur.right

        # 找出字典中最大的key
        res = []
        for key, value in dist.items():
            if (value == max(dist.values())):
                res.append(key)
        return res

# 迭代法-中序遍历-不使用额外空间，利用二叉搜索树特性：
class Solution:
    def findMode(self, root: TreeNode) -> List[int]:
        stack = []
        cur = root
        pre = None
        maxCount, count = 0, 0
        res = []
        while cur or stack:
            if cur:  # 指针来访问节点，访问到最底层
                stack.append(cur)
                cur = cur.left
            else:  # 逐一处理节点
                cur = stack.pop()
                if pre == None:  # 第一个节点
                    count = 1
                elif pre.val == cur.val:  # 与前一个节点数值相同
                    count += 1
                else:
                    count = 1
                if count == maxCount:
                    res.append(cur.val)
                if count > maxCount:
                    maxCount = count
                    res.clear()
                    res.append(cur.val)

                pre = cur
                cur = cur.right
        return res	
```



#### 404. Sum of Left Leaves

```python
# 递归
class Solution:
    def sumOfLeftLeaves(self, root: Optional[TreeNode]) -> int:
        if not root:
            return 0
        left = self.sumOfLeftLeaves(root.left)
        right = self.sumOfLeftLeaves(root.right)
        leaf = 0
        if root.left and not root.left.left and not root.left.right:
            leaf = root.left.val
        return leaf + left + right
```

```python
# 迭代
class Solution:
    def sumOfLeftLeaves(self, root: TreeNode) -> int:
        stack = []
        if root: 
            stack.append(root)
        res = 0
        while stack: 
            # 每次都把当前节点的左节点加进去. 
            cur_node = stack.pop()
            if cur_node.left and not cur_node.left.left and not cur_node.left.right: 
                res += cur_node.left.val
            if cur_node.left: 
                stack.append(cur_node.left)
            if cur_node.right: 
                stack.append(cur_node.right)
        return res
```



#### 513. Find Bottom Left Tree Value

```python
# 迭代（层序遍历BFS）
class Solution:
    def findBottomLeftValue(self, root: Optional[TreeNode]) -> int:
        if not root:
            return
        queue = collections.deque([root])
        ans = root.val
        while queue:
            n = len(queue)
            for i in range(n):
                curr = queue.popleft()
                if i== 0: ans = curr.val
                if curr.left: queue.append(curr.left)
                if curr.right: queue.append(curr.right)
        return ans
```

```python
# 递归（回溯）
class Solution:
    def findBottomLeftValue(self, root: TreeNode) -> int:
        self.max_depth = -float("INF")
        self.leftmost_val = 0

        def __traverse(root, cur_depth): 
            if not root.left and not root.right: 
                if cur_depth > self.max_depth: 
                    self.max_depth = cur_depth
                    self.leftmost_val = root.val  
            if root.left: 
                cur_depth += 1
                __traverse(root.left, cur_depth)
                cur_depth -= 1
            if root.right: 
                cur_depth += 1
                __traverse(root.right, cur_depth)
                cur_depth -= 1
                
        __traverse(root, 0)
        return self.leftmost_val
```



---

#### 331. Verify Preorder Serialization of a Binary Tree

One way to serialize a binary tree is to use **preorder traversal**. When we encounter a non-null node, we record the node's value. If it is a null node, we record using a sentinel value such as `'#'`.

![img](https://assets.leetcode.com/uploads/2021/03/12/pre-tree.jpg)

For example, the above binary tree can be serialized to the string `"9,3,4,#,#,1,#,#,2,#,6,#,#"`, where `'#'` represents a null node.

Given a string of comma-separated values `preorder`, return `true` if it is a correct preorder traversal serialization of a binary tree.

It is **guaranteed** that each comma-separated value in the string must be either an integer or a character `'#'` representing null pointer.

You may assume that the input format is always valid.

- For example, it could never contain two consecutive commas, such as `"1,,3"`.

**Note:** You are not allowed to reconstruct the tree.

 

**Example 1:**

```
Input: preorder = "9,3,4,#,#,1,#,#,2,#,6,#,#"
Output: true
```

```python
class Solution(object):
    def isValidSerialization(self, preorder):
        stack = []
        for node in preorder.split(','):
            stack.append(node)
            while len(stack) >= 3 and stack[-1] == stack[-2] == '#' and stack[-3] != '#':
                stack.pop(), stack.pop(), stack.pop()
                stack.append('#')
        return len(stack) == 1 and stack.pop() == '#'
```

```python
class Solution:
    def isValidSerialization(self, preorder: str) -> bool:
        diff = 1
        # diff = 出度 - 入度
        for node in preorder.split(','):
            diff -= 1
            if diff < 0:
                return False
            if node != '#':
                diff += 2
        return diff == 0
```





#### 653. Two Sum IV - Input is a BST

Given the `root` of a Binary Search Tree and a target number `k`, return *`true` if there exist two elements in the BST such that their sum is equal to the given target*.

 

**Example 1:**

![img](https://assets.leetcode.com/uploads/2020/09/21/sum_tree_1.jpg)

```
Input: root = [5,3,6,2,4,null,7], k = 9
Output: true
```

```python
# hashmap
class Solution:
    def findTarget(self, root: Optional[TreeNode], k: int) -> bool:
        def find(root, k, h):
            if not root:
                return False
            if k - root.val in h:
                return True
            h.add(root.val)
            return find(root.left, k, h) or find(root.right, k, h) 
        
        h = {root.val}
        return find(root, k, h)
```







### Board

#### 36. Valid Sudoku

Determine if a `9 x 9` Sudoku board is valid. Only the filled cells need to be validated **according to the following rules**:

1. Each row must contain the digits `1-9` without repetition.
2. Each column must contain the digits `1-9` without repetition.
3. Each of the nine `3 x 3` sub-boxes of the grid must contain the digits `1-9` without repetition.

**Note:**

- A Sudoku board (partially filled) could be valid but is not necessarily solvable.
- Only the filled cells need to be validated according to the mentioned rules.

 

**Example 1:**

![img](https://upload.wikimedia.org/wikipedia/commons/thumb/f/ff/Sudoku-by-L2G-20050714.svg/250px-Sudoku-by-L2G-20050714.svg.png)

```
Input: board = 
[["5","3",".",".","7",".",".",".","."]
,["6",".",".","1","9","5",".",".","."]
,[".","9","8",".",".",".",".","6","."]
,["8",".",".",".","6",".",".",".","3"]
,["4",".",".","8",".","3",".",".","1"]
,["7",".",".",".","2",".",".",".","6"]
,[".","6",".",".",".",".","2","8","."]
,[".",".",".","4","1","9",".",".","5"]
,[".",".",".",".","8",".",".","7","9"]]
Output: true
```

```python
class Solution:
    def isValidSudoku(self, board):
        """
        :type board: List[List[str]]
        :rtype: bool
        """
        # init data
        rows = [{} for i in range(9)]
        columns = [{} for i in range(9)]
        boxes = [{} for i in range(9)]

        # validate a board
        for i in range(9):
            for j in range(9):
                num = board[i][j]
                if num != '.':
                    num = int(num)
                    box_index = (i // 3 ) * 3 + j // 3
                    
                    # keep the current cell value
                    rows[i][num] = rows[i].get(num, 0) + 1
                    columns[j][num] = columns[j].get(num, 0) + 1
                    boxes[box_index][num] = boxes[box_index].get(num, 0) + 1
                    
                    # check if this value has been already seen before
                    if rows[i][num] > 1 or columns[j][num] > 1 or boxes[box_index][num] > 1:
                        return False         
        return True
```



#### 54. Spiral Matrix

Given an `m x n` `matrix`, return *all elements of the* `matrix` *in spiral order*.

 

**Example 1:**

![img](https://assets.leetcode.com/uploads/2020/11/13/spiral1.jpg)

```
Input: matrix = [[1,2,3],[4,5,6],[7,8,9]]
Output: [1,2,3,6,9,8,7,4,5]
```

```python
class Solution(object):
    def spiralOrder(self, matrix):
        def spiral_coords(r1, c1, r2, c2):
            for c in range(c1, c2 + 1):
                yield r1, c
            for r in range(r1 + 1, r2 + 1):
                yield r, c2
            if r1 < r2 and c1 < c2:
                for c in range(c2 - 1, c1, -1):
                    yield r2, c
                for r in range(r2, r1, -1):
                    yield r, c1

        if not matrix: return []
        ans = []
        r1, r2 = 0, len(matrix) - 1
        c1, c2 = 0, len(matrix[0]) - 1
        while r1 <= r2 and c1 <= c2:
            for r, c in spiral_coords(r1, c1, r2, c2):
                ans.append(matrix[r][c])
            r1 += 1; r2 -= 1
            c1 += 1; c2 -= 1
        return ans
```



#### 59. Spiral Matrix II

Given a positive integer `n`, generate an `n x n` `matrix` filled with elements from `1` to `n2` in spiral order.

 

**Example 1:**

![img](https://assets.leetcode.com/uploads/2020/11/13/spiraln.jpg)

```
Input: n = 3
Output: [[1,2,3],[8,9,4],[7,6,5]]
```



```python
class Solution:
    def generateMatrix(self, n: int) -> List[List[int]]:
        left, right, up, down = 0, n-1, 0, n-1
        matrix = [ [0]*n for _ in range(n)]
        num = 1
        while left<=right and up<=down:
            # 填充左到右
            for i in range(left, right+1):
                matrix[up][i] = num
                num += 1
            up += 1
            # 填充上到下
            for i in range(up, down+1):
                matrix[i][right] = num
                num += 1
            right -= 1
            # 填充右到左
            for i in range(right, left-1, -1):
                matrix[down][i] = num
                num += 1
            down -= 1
            # 填充下到上
            for i in range(down, up-1, -1):
                matrix[i][left] = num
                num += 1
            left += 1
        return matrix
```



```python
class Solution:
    def generateMatrix(self, n: int) -> List[List[int]]:
        matrix = [[0 for i in range(n)] for j in range(n)]
        i = 0
        j = 0
        direct = 0
        index = 1
        while index <= n**2:
            matrix[i][j] = index
            index += 1
            direct = self.move(direct, matrix, i, j, n)
            if direct == 1:
                j += 1
            elif direct == 2:
                i += 1
            elif direct == 3:
                j -= 1
            else:
                i -= 1
            
        return matrix
            
    def move(self, num, matrix, i, j, n):
        if self.try_move(num, matrix, i, j, n):
            return num
        return (num + 1) % 4
        
    def try_move(self, num, matrix, i, j, n):
        if num == 0 and i > 0 and matrix[i-1][j] == 0:
            return True
        if num == 1 and j < n-1 and matrix[i][j+1] == 0:
            return True
        if num == 2 and i < n-1 and matrix[i+1][j] == 0:
            return True
        if num == 3 and j > 0 and matrix[i][j-1] == 0:
            return True
        return False
            
```

#### 200. Number of Islands

```python
# DFS
class Solution:
    def dfs(self, grid, r, c):
        grid[r][c] = 0
        nr, nc = len(grid), len(grid[0])
        for x, y in [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]:
            if 0 <= x < nr and 0 <= y < nc and grid[x][y] == "1":
                self.dfs(grid, x, y)

    def numIslands(self, grid: List[List[str]]) -> int:
        nr = len(grid)
        if nr == 0:
            return 0
        nc = len(grid[0])

        num_islands = 0
        for r in range(nr):
            for c in range(nc):
                if grid[r][c] == "1":
                    num_islands += 1
                    self.dfs(grid, r, c)
        
        return num_islands
```

```python
# BFS
class Solution:
    def numIslands(self, grid: List[List[str]]) -> int:
        nr = len(grid)
        if nr == 0:
            return 0
        nc = len(grid[0])

        num_islands = 0
        for r in range(nr):
            for c in range(nc):
                if grid[r][c] == "1":
                    num_islands += 1
                    grid[r][c] = "0"
                    neighbors = collections.deque([(r, c)])
                    while neighbors:
                        row, col = neighbors.popleft()
                        for x, y in [(row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)]:
                            if 0 <= x < nr and 0 <= y < nc and grid[x][y] == "1":
                                neighbors.append((x, y))
                                grid[x][y] = "0"
        
        return num_islands
```

#### 240. Search a 2D Matrix II

```python
class Solution:
    def searchMatrix(self, matrix, target):
        if len(matrix) == 0 or len(matrix[0]) == 0:
            return False

        height = len(matrix)
        width = len(matrix[0])

        # start our "pointer" in the bottom-left
        row = height-1
        col = 0

        while col < width and row >= 0:
            if matrix[row][col] > target:
                row -= 1
            elif matrix[row][col] < target:
                col += 1
            else: # found it
                return True
        
        return False
```



### Math

#### 204. Count Primes

Count the number of prime numbers less than a non-negative number, `n`.

 

**Example 1:**

```
Input: n = 10
Output: 4
Explanation: There are 4 prime numbers less than 10, they are 2, 3, 5, 7.
```

```python
# 埃氏筛
class Solution:
    def countPrimes(self, n: int) -> int:
        # 定义数组标记是否是质数
        is_prime = [1] * n
        count = 0
        for i in range(2, n):
            # 将质数的倍数标记为合数
            if is_prime[i]:
                count += 1
                # 从 i*i 开始标记
                for j in range(i*i, n, i):
                    is_prime[j] = 0
        return count
```

```python
# 线性筛
class Solution:
    def countPrimes(self, n: int) -> int:
        if n < 2:
            return 0
        count = 0
        is_prime = [1] * n
        prime = []
        for i in range(2,n,1):
            if is_prime[i]:
                prime.append(i)
                count += 1
            j = 0
            while j < count and i * prime[j] < n:
                is_prime[i * prime[j]] = False
                #如果已经找到了i的质因子（如果i是合数） 就不再继续
                if i % prime[j] == 0:
                    break
                j += 1
        return count
```

#### 264. Ugly Number II

An **ugly number** is a positive integer whose prime factors are limited to `2`, `3`, and `5`.

Given an integer `n`, return *the* `nth` ***ugly number***.

 

**Example 1:**

```
Input: n = 10
Output: 12
Explanation: [1, 2, 3, 4, 5, 6, 8, 9, 10, 12] is the sequence of the first 10 ugly numbers.
```

```python
class Solution:
    def nthUglyNumber(self, n: int) -> int:
        factors = [2, 3, 5]
        seen = {1}
        heap = [1]
        
        for i in range(n - 1):
            curr = heapq.heappop(heap)
            for factor in factors:
                if (nxt := curr * factor) not in seen:
                    seen.add(nxt)
                    heapq.heappush(heap, nxt)
                    
        return heapq.heappop(heap)
```

```python
class Solution:
    def nthUglyNumber(self, n: int) -> int:
        dp = [0] * (n + 1)
        dp[1] = 1
        p2 = p3 = p5 = 1

        for i in range(2, n + 1):
            num2, num3, num5 = dp[p2] * 2, dp[p3] * 3, dp[p5] * 5
            dp[i] = min(num2, num3, num5)
            if dp[i] == num2:
                p2 += 1
            if dp[i] == num3:
                p3 += 1
            if dp[i] == num5:
                p5 += 1
        
        return dp[n]
```



#### 313. Super Ugly Number

A **super ugly number** is a positive integer whose prime factors are in the array `primes`.

Given an integer `n` and an array of integers `primes`, return *the* `nth` ***super ugly number***.

The `nth` **super ugly number** is **guaranteed** to fit in a **32-bit** signed integer.

 

**Example 1:**

```
Input: n = 12, primes = [2,7,13,19]
Output: 32
Explanation: [1,2,4,7,8,13,14,16,19,26,28,32] is the sequence of the first 12 super ugly numbers given primes = [2,7,13,19].
```

```python
class Solution:
    def nthSuperUglyNumber(self, n: int, primes: List[int]) -> int:
        seen = {1}
        heap = [1]
        
        for i in range(n-1):
            curr = heapq.heappop(heap)
            for factor in primes:
                if (nxt := factor * curr) not in seen:
                    seen.add(nxt)
                    heapq.heappush(heap, nxt)
                    
        return heapq.heappop(heap)
```

```python
class Solution:
    def nthSuperUglyNumber(self, n: int, primes: List[int]) -> int:
        dp = [0] * (n + 1)
        dp[1] = 1
        m = len(primes)
        pointers = [1] * m

        for i in range(2, n + 1):
            min_num = min(dp[pointers[j]] * primes[j] for j in range(m))
            dp[i] = min_num
            for j in range(m):
                if dp[pointers[j]] * primes[j] == min_num:
                    pointers[j] += 1
        
        return dp[n]
```

#### 633. Sum of Square Numbers

```python
class Solution(object):
    def judgeSquareSum(self, c):
        """
        :type c: int
        :rtype: bool
        """
        if c == 0: return True
        for a in range(1, int(math.sqrt(c) + 1)):
            b = c - a * a
            if int(math.sqrt(b)) ** 2 == b:
                return True
        return False
```

```python
class Solution:
    def judgeSquareSum(self, c: int) -> bool:
        if not c:
            return True
        # (a - b) ^ 2 + (a + b) ^ 2 = 2 * (a ^ 2 + b ^ 2) = 2 * c
        while c % 2 == 0:
            c //= 2
        # 费马平方和定理
        if c % 4 == 3:
            return False
        sqrt = int(math.sqrt(c))
        for i in range(3, sqrt + 1, 4):
            count = 0
            while c % i == 0:
                c //= i
                count += 1
            if count % 2 != 0:
                return False
        return True
```

### Graph

#### 207. Course Schedule

BFS

```python
class Solution:
    def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
        # 存储有向图
        edges = collections.defaultdict(list)
        # 存储每个节点入度
        indeg = [0] * numCourses
        
        for course in prerequisites:
            edges[course[1]].append(course[0])
            indeg[course[0]] += 1
            
        q = collections.deque([u for u in range(numCourses) if indeg[u] == 0])
        visited = 0
        
        while q:
            curr = q.popleft()
            visited += 1
            for v in edges[curr]:
                indeg[v] -= 1
                if indeg[v] == 0:
                    q.append(v)
        print(visited)
        return visited == numCourses
```

DFS

```python
class Solution:
    def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
        edges = collections.defaultdict(list)
        visited = [0] * numCourses
        #[未搜索0]：我们还没有搜索到这个节点；
        #[搜索中1]：我们搜索过这个节点，但还没有回溯到该节点，即该节点还没有入栈，还有相邻的节点没有搜索完成）；
        #[已完成2]：我们搜索过并且回溯过这个节点，即该节点已经入栈，并且所有该节点的相邻节点都出现在栈的更底部的位置，满足拓扑排序的要求。

        result = list()
        valid = True

        for info in prerequisites:
            edges[info[1]].append(info[0])
        
        def dfs(u: int):
            nonlocal valid
            visited[u] = 1
            for v in edges[u]:
                if visited[v] == 0:
                    dfs(v)
                    if not valid:
                        return
                elif visited[v] == 1:
                    valid = False
                    return
            visited[u] = 2
            result.append(u)
        
        for i in range(numCourses):
            if valid and not visited[i]:
                dfs(i)
        
        return valid
```

#### 210. Course Schedule II

DFS

```python
class Solution:
    def findOrder(self, numCourses: int, prerequisites: List[List[int]]) -> List[int]:
        # 存储有向图
        edges = collections.defaultdict(list)
        # 标记每个节点的状态：0=未搜索，1=搜索中，2=已完成
        visited = [0] * numCourses
        # 用数组来模拟栈，下标 0 为栈底，n-1 为栈顶
        result = list()
        # 判断有向图中是否有环
        valid = True
        
        for info in prerequisites:
            edges[info[0]].append(info[1]) # 如果info[1]->info[0]则最后return result[::-1]
        
        def dfs(u: int):
            nonlocal valid
            # 将节点标记为「搜索中」
            visited[u] = 1
            # 搜索其相邻节点
            # 只要发现有环，立刻停止搜索
            for v in edges[u]:
                # 如果「未搜索」那么搜索相邻节点
                if visited[v] == 0:
                    dfs(v)
                    if not valid:
                        return
                # 如果「搜索中」说明找到了环
                elif visited[v] == 1:
                    valid = False
                    return
            # 将节点标记为「已完成」
            visited[u] = 2
            # 将节点入栈
            result.append(u)
        
        # 每次挑选一个「未搜索」的节点，开始进行深度优先搜索
        for i in range(numCourses):
            if valid and not visited[i]:
                dfs(i)
        
        if not valid:
            return list()
        
        # 如果没有环，那么就有拓扑排序
        # 注意下标 0 为栈底，因此需要将数组反序输出
        return result
```

BFS

```python
class Solution:
    def findOrder(self, numCourses: int, prerequisites: List[List[int]]) -> List[int]:
        # 存储有向图
        edges = collections.defaultdict(list)
        # 存储每个节点的入度
        indeg = [0] * numCourses
        # 存储答案
        result = list()

        for info in prerequisites:
            edges[info[1]].append(info[0])
            indeg[info[0]] += 1
        
        # 将所有入度为 0 的节点放入队列中
        q = collections.deque([u for u in range(numCourses) if indeg[u] == 0])

        while q:
            # 从队首取出一个节点
            u = q.popleft()
            # 放入答案中
            result.append(u)
            for v in edges[u]:
                indeg[v] -= 1
                # 如果相邻节点 v 的入度为 0，就可以选 v 对应的课程了
                if indeg[v] == 0:
                    q.append(v)

        if len(result) != numCourses:
            result = list()
        return result
```



#### 565. Array Nesting

```mysql
# mysolution
class Solution:
    def arrayNesting(self, nums: List[int]) -> int:
        result = 1
        for i in range(len(nums)):
            if nums[i] != -1: # not visited
                length = 0
                temp = nums[i]
                nums[i] = -1
                while temp != -1:
                    pos = temp
                    temp = nums[temp]
                    nums[pos] = -1
                    length += 1

                result = max(result, length)
        return result
```

```mysql
class Solution:
    def arrayNesting(self, nums: List[int]) -> int:  
        cache = {}

        def dfs(i, visited):
            if nums[i] in visited:
                return 0
            if i in cache: return cache[i]
            visited.add(nums[i])
            ans = 1 + dfs(nums[i], visited)
            cache[i] = ans
            return ans

        res = 0
        for i in range(len(nums)):
            res = max(res, dfs(i, set()))
        return res
```

```mysql
class Solution:
    def arrayNesting(self, nums: List[int]) -> int:  
        n = len(nums)
        visited = [0] * n
        res = 0
        for i in range(n):
            cur, cnt = i, 0
            while not visited[cur]:
                visited[cur] = 1
                cur = nums[cur]
                cnt += 1
            res = max(res, cnt)
        return res
```

#### 并查集

```python
class UnionFind:
    def __init__(self, size):
        self.root = [i for i in range(size)]
        # Use a rank array to record the height of each vertex, i.e., the "rank" of each vertex.
        # The initial "rank" of each vertex is 1, because each of them is
        # a standalone vertex with no connection to other vertices.
        self.rank = [1] * size

    # The find function here is the same as that in the disjoint set with path compression.
    def find(self, x):
        if x == self.root[x]:
            return x
        self.root[x] = self.find(self.root[x])
        return self.root[x]

    # The union function with union by rank
    def union(self, x, y):
        rootX = self.find(x)
        rootY = self.find(y)
        if rootX != rootY:
            if self.rank[rootX] > self.rank[rootY]:
                self.root[rootY] = rootX
            elif self.rank[rootX] < self.rank[rootY]:
                self.root[rootX] = rootY
            else:
                self.root[rootY] = rootX
                self.rank[rootX] += 1

    def connected(self, x, y):
        return self.find(x) == self.find(y)
```



```python
class UnionFind:
    def __init__(self):
        self.father = {}
    
    def find(self,x):
        root = x

        while self.father[root] != None:
            root = self.father[root]

        while x != root:
            original_father = self.father[x]
            self.father[x] = root
            x = original_father
         
        return root
    
    def merge(self,x,y,val):
        root_x,root_y = self.find(x),self.find(y)
        if root_x != root_y:
            self.father[root_x] = root_y

    def is_connected(self,x,y):
        return self.find(x) == self.find(y)
    
    def add(self,x):
        if x not in self.father:
            self.father[x] = None
```

#### 547. Number of Province

#### 323. Number of Connected Components in an Undirected Graph

```python
class UnionFind():
    def __init__(self, n):
        self.father = [i for i in range(n)]
        self.size = [1] * n
        self.count = n
        
    def find(self, x):
        while x != self.father[x]:
            # self.father[x] = self.find(self.father[x]) # 压缩路径
            x = self.father[x]
        return self.father[x]

    def union(self, x, y):
        father_x, father_y = self.find(x), self.find(y)
        if father_x != father_y:
            # if self.size[father_x] > self.size[father_y]:
            #     father_x, father_y = father_y, father_x
            self.father[father_x] = father_y
            self.size[father_y] += self.size[father_x]
            self.count -= 1

class Solution:
    def countComponents(self, n: int, edges: List[List[int]]) -> int:
        uf = unionfind(n)
                
        for x, y in edges:
            uf.union(x, y)
        return uf.count

```

```python
class unionfind():
    def __init__(self,n):
        self.father={}
        for i in range(n): self.father[i]=None
    def find(self, i):
        root=i
        while self.father[root]!=None:
            root=self.father[root]
        return root
    def union(self,u, v):
        root1, root2=self.find(u), self.find(v)
        if root1!=root2:
            self.father[root1]=root2
            
def countComponents(n, edges):
    uf=unionfind(n)
    groups=n
    for u,v in edges:
        if uf.find(u)!=uf.find(v):
            groups-=1
            uf.union(u, v)
    return groups
```

#### 261. Graph Valid Tree

```python
class unionfind():
    def __init__(self, n):
        self.father = [i for i in range(n)]
        self.size = [1] * n
        self.count = n
        
    def find(self, x):
        while x != self.father[x]:
            # self.father[x] = self.find(father[x]) # 压缩路径
            x = self.father[x]
        return self.father[x]

    def union(self, x, y):
        father_x, father_y = self.find(x), self.find(y)
        if father_x == father_y: return True
        if father_x != father_y:
            # if self.size[father_x] > self.size[father_y]:
            #     father_x, father_y = father_y, father_x
            self.father[father_x] = father_y
            self.size[father_y] += self.size[father_x]
            self.count -= 1
            return False
       
class Solution:
    def validTree(self, n: int, edges: List[List[int]]) -> bool:
        uf = unionfind(n)
        for x, y in edges:
            if uf.union(x, y):
                return False
        
        return True if uf.count == 1 else False
```

#### DFS

```PYTHON
def dfs(start, end):
    if SOMETHING
    if start == end:
        return
    visited.add(start)
    for next_move in graph[start]:
        if next_move not in visited:
            dfs(next_move, end)
```



#### 797. All Paths From Source to Target

```python
class Solution:
    def allPathsSourceTarget(self, graph: List[List[int]]) -> List[List[int]]:
        def dfs(start, end, path):
            if start == end:
                result.append(path)
                return 
                
            for i in graph[start]:
                dfs(i, end, path + [i])
        n = len(graph)
        path = [0]
        result = []
        dfs(0, n-1, path)
        
        return result
```



#### 1971. Find if Path Exists in Graph

```python
class Solution:
    def validPath(self, n: int, edges: List[List[int]], start: int, end: int) -> bool:
        if start == end:
            return True
        graph = collections.defaultdict(list)
        for x,y in edges:
            graph[x].append(y)
            graph[y].append(x)

        queue = [start]
        visited = {start}
        while queue:
            
            p = queue.pop(0)
            if p == end:
                return True
            for q in graph[p]:
                if q not in visited:
                    visited.add(q)
                    queue.append(q)
        return False
```

```python
#dfs
class Solution:
    def validPath(self, n: int, edges: List[List[int]], start: int, end: int) -> bool:
        
        def dfs(start, end):
            if self.found == True:
                return
            if start == end:
                self.found = True
                return
            visited.add(start)
            for next_move in graph[start]:
                if next_move not in visited:
                    dfs(next_move, end)
                    
        if start == end: return True
        
        self.found = False
        visited = set()
        graph = defaultdict(list)
        
        for x,y in edges:
            graph[x].append(y)
            graph[y].append(x)
            
        dfs(start, end)
        return self.found
```

#### 133. Clone Graph

#### BFS

我们都知道 BFS 需要使用队列，代码框架是这样子的（伪代码）：

```python
while queue 非空:
	node = queue.pop()
    for node 的所有相邻结点 m:
        if m 未访问过:
            queue.push(m)
```

但是用 BFS 来求最短路径的话，这个队列中第 1 层和第 2 层的结点会紧挨在一起，无法区分。因此，我们需要稍微修改一下代码，在每一层遍历开始前，记录队列中的结点数量 nn ，然后一口气处理完这一层的 nn 个结点。代码框架是这样的：

```python

depth = 0 # 记录遍历到第几层
while queue 非空:
    depth++
    n = queue 中的元素个数
    循环 n 次:
        node = queue.pop()
        for node 的所有相邻结点 m:
            if m 未访问过:
                queue.push(m)
```



#### 1059. All Paths from Source Lead to Destination

```python
class Solution:
    def leadsToDestination(self, n: int, edges: List[List[int]], source: int, destination: int) -> bool:
        self.result = True
        graph = defaultdict(list)
        for x,y in edges:
            graph[x].append(y)
        visited = [0] * n
        
        def dfs(node, result):
            if visited[node] != 0:
                self.result = (visited[node] == 2)
                return 
            if len(graph[node]) == 0:
                self.result = (node == destination)
                return 
            visited[node] = 1
            for i in graph[node]:
                dfs(i, self.result)
                if not self.result:
                    return
            visited[node] = 2
            
        dfs(source, self.result)
        
        return self.result
```

```python
class Solution:
    def allPathsSourceTarget(self, graph: List[List[int]]) -> List[List[int]]:
        n = len(graph)
        result = []
        queue = [[0]]
        while queue:
            p = queue.pop(0)
            if p[-1] == n - 1:
                result.append(p)
            for q in graph[p[-1]]:
                queue.append(p + [q])
        return result
```

#### 1091. Shortest Path in Binary Matrix

```python
class Solution:
    
    def shortestPathBinaryMatrix(self, grid: List[List[int]]) -> int:  
        
        max_row = len(grid) - 1
        max_col = len(grid[0]) - 1
        directions = [
            (-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        
        # Helper function to find the neighbors of a given cell.
        def get_neighbours(row, col):
            for row_difference, col_difference in directions:
                new_row = row + row_difference
                new_col = col + col_difference
                if not(0 <= new_row <= max_row and 0 <= new_col <= max_col):
                    continue
                if grid[new_row][new_col] != 0:
                    continue
                yield (new_row, new_col)
        
        # Check that the first and last cells are open. 
        if grid[0][0] != 0 or grid[max_row][max_col] != 0:
            return -1
        
        # Set up the BFS.
        queue = deque([(0, 0)])
        visited = {(0, 0)}
        current_distance = 1
        
        # Do the BFS.
        while queue:
            # Process all nodes at current_distance from the top-left cell.
            nodes_of_current_distance = len(queue)
            for _ in range(nodes_of_current_distance):
                row, col = queue.popleft()
                if (row, col) == (max_row, max_col):
                    return current_distance
                for neighbour in get_neighbours(row, col):
                    if neighbour in visited:
                        continue
                    visited.add(neighbour)
                    queue.append(neighbour)
            # We'll now be processing all nodes at current_distance + 1
            current_distance += 1
                    
        # There was no path.
        return -1 
```

#### 994. Rotting Oranges

```PYTHON
class Solution:
    def orangesRotting(self, grid: List[List[int]]) -> int:
        M = len(grid)
        N = len(grid[0])
        queue = []
        direc = [(-1, 0), (0, 1), (1, 0), (0, -1)]
    
        def neighbors(row, col):
            for x, y in direc:
                new_row, new_col = row + x, col + y
                if M > new_row >= 0 and N > new_col >= 0:
                    yield (new_row, new_col)

        count = 0 # count 表示新鲜橘子的数量
        for r in range(M):
            for c in range(N):
                if grid[r][c] == 1:
                    count += 1
                elif grid[r][c] == 2:
                    queue.append((r, c))

        round = 0 # round 表示腐烂的轮数，或者分钟数
        while count > 0 and len(queue) > 0:
            round += 1 
            n = len(queue)
            for i in range(n):
                r, c = queue.pop(0)
                for x, y in neighbors(r, c):
                    if grid[x][y] == 1:
                        grid[x][y] = 2
                        count -= 1
                        queue.append((x, y))


        if count > 0:
            return -1
        else:
            return round
```

#### Minimal Spanning Tree

#### 1584. Min Cost to Connect All Points

Kruskal Algorithm

![image-20210906095551742](C:\Users\zeban\AppData\Roaming\Typora\typora-user-images\image-20210906095551742.png)

```python
class UnionFind():
    def __init__(self, n):
        self.father = [i for i in range(n)]
        self.size = [1] * n
        self.count = n
        
    def find(self, x):
        while x != self.father[x]:
            self.father[x] = self.find(self.father[x]) # 压缩路径
            x = self.father[x]
        return self.father[x]

    def union(self, x, y):
        father_x, father_y = self.find(x), self.find(y)
        if father_x != father_y:
            if self.size[father_x] > self.size[father_y]:
                father_x, father_y = father_y, father_x
            self.father[father_x] = father_y
            self.size[father_y] += self.size[father_x]
            self.count -= 1
    def is_connected(self,x, y):
        return self.find(x) == self.find(y)

class Solution:
    def minCostConnectPoints(self, points: List[List[int]]) -> int:
        dist = lambda x, y: abs(points[x][0] - points[y][0]) + abs(points[x][1] - points[y][1])

        n = len(points)
        u = UnionFind(n)
        edges = list()

        for i in range(n):
            for j in range(i + 1, n):
                edges.append((dist(i, j), i, j))
        
        edges.sort()
        
        result, num = 0, 1
        for length, x, y in edges:
            if not u.is_connected(x, y):
                u.union(x, y)
                result += length
                num += 1
                if num == n:
                    break
        
        return result

```

Prim Algorithm

![image-20210906095857339](C:\Users\zeban\AppData\Roaming\Typora\typora-user-images\image-20210906095857339.png)

```python
class Solution:
    def minCostConnectPoints(self, points: List[List[int]]) -> int:
        #Prim算法
        n = len(points)
        d = [float("inf")] * n # 表示各个顶点与加入最小生成树的顶点之间的最小距离.
        vis = [False] * n # 表示是否已经加入到了最小生成树里面
        d[0] = 0
        ans = 0
        for _ in range(n):
            # 寻找目前这轮的最小d
            M = float("inf") 
            for i in range(n):
                if not vis[i] and d[i] < M:
                    node = i
                    M = d[i]
            vis[node] = True
            ans += M
            # 更新与这个顶点相连接的顶点的d.
            for i in range(n):
                if not vis[i]:
                    d[i] = min(d[i], abs(points[i][0] - points[node][0]) + abs(points[i][1] - points[node][1]))
        return ans
```

#### Dijkstra's Algorithm

#### 743. Network Delay Time

Dijkstra

```python
class Solution:
    def networkDelayTime(self, times: List[List[int]], n: int, k: int) -> int:
        g = [[] for _ in range(n)]
        for x, y, time in times:
            g[x - 1].append((y - 1, time))

        dist = [float('inf')] * n
        dist[k - 1] = 0
        q = [(0, k - 1)]
        while q:
            time, x = heapq.heappop(q)
            if dist[x] < time:
                continue
            for y, time in g[x]:
                if (d := dist[x] + time) < dist[y]:
                    dist[y] = d
                    heapq.heappush(q, (d, y))

        ans = max(dist)
        return ans if ans < float('inf') else -1
```



BFS

```python
class Solution:
    def networkDelayTime(self, times: List[List[int]], n: int, k: int) -> int:
        graph = defaultdict(list)
        for u, v, w in times:
            graph[u].append((v, w))
        
        dist = [float('inf')] * n
        
        def dfs(p, time):
            if time >= dist[p-1]: return
            dist[p-1] = time
            
            for v, w in sorted(graph[p], key=lambda x: x[1]):
                dfs(v, time + w)
        
        dfs(k, 0)
        ans = max(dist)
        return ans if ans < float('inf') else -1
```

#### Bellman-Ford

#### 787. Cheapest Flights Within K Stops

```python
class Solution:
    def findCheapestPrice(self, n: int, flights: List[List[int]], src: int, dst: int, k: int) -> int:
        p = [float("inf")] * n
        p[src] = 0
        c = p.copy()
        ans = float("inf")
        for t in range(1, k + 2):
            for j, i, cost in flights:
                c[i] = min(c[i], p[j] + cost)
            p = c.copy()
            ans = min(ans, p[dst])
        
        return -1 if ans == float("inf") else ans
```



### Divide

#### 215. Kth Largest Element in an Array

堆排序

```python
class Solution:
    def findKthLargest(self, nums: List[int], k: int) -> int:
        heap = []
        for num in nums:
            heapq.heappush(heap, num)
            if len(heap) > k:
                heapq.heappop(heap)
        return heap[0]
```

Partition函数

```python
def partition(nums, left, right):
    pivot = nums[left]#初始化一个待比较数据
    i,j = left, right
    while(i < j):
        while(i<j and nums[j]>=pivot): #从后往前查找，直到找到一个比pivot更小的数
            j-=1
        nums[i] = nums[j] #将更小的数放入左边
        while(i<j and nums[i]<=pivot): #从前往后找，直到找到一个比pivot更大的数
            i+=1
        nums[j] = nums[i] #将更大的数放入右边
    #循环结束，i与j相等
    nums[i] = pivot #待比较数据放入最终位置 
    return i #返回待比较数据最终位置
```

快速排序

```python
#快速排序
def quicksort(nums, left, right):
    if left < right:
        index = partition(nums, left, right)
        quicksort(nums, left, index-1)
        quicksort(nums, index+1, right)

arr = [1,3,2,2,0]
quicksort(arr, 0, len(arr)-1)
print(arr)
```

topk切分: 将快速排序改成快速选择，即我们希望寻找到一个位置，这个位置左边是k个比这个位置上的数更小的数，右边是n-k个比该位置上的数大的数，我将它命名为topk_split，找到这个位置后停止迭代，完成了一次划分。

```python
def topk_split(nums, k, left, right):
    #寻找到第k个数停止递归，使得nums数组中index左边是前k个小的数，index右边是后面n-k个大的数
    if (left<right):
        index = partition(nums, left, right)
        if index==k:
            return 
        elif index < k:
            topk_split(nums, k, index+1, right)
        else:
            topk_split(nums, k, left, index-1)
```

接下来就依赖于上面这两个函数解决所有的topk问题.

获得前k小的数

```python
#获得前k小的数
def topk_smalls(nums, k):
    topk_split(nums, k, 0, len(nums)-1)
    return nums[:k]

arr = [1,3,2,3,0,-19]
k = 2
print(topk_smalls(arr, k))
print(arr)
```

获取第k小的数（略）

获得前k大的数

```python
#获得前k大的数 
def topk_larges(nums, k):
    #parttion是按从小到大划分的，如果让index左边为前n-k个小的数，则index右边为前k个大的数
    topk_split(nums, len(nums)-k, 0, len(nums)-1) #把k换成len(nums)-k
    return nums[len(nums)-k:] 

arr = [1,3,-2,3,0,-19]
k = 3
print(topk_larges(arr, k))
print(arr)
```

获得第k大的数（略）

只排序前k个小的数

```python
#只排序前k个小的数
#获得前k小的数O(n)，进行快排O(klogk)
def topk_sort_left(nums, k):
    topk_split(nums, k, 0, len(nums)-1) 
    topk = nums[:k]
    quicksort(topk, 0, len(topk)-1)
    return topk+nums[k:] #只排序前k个数字

arr = [0,0,1,3,4,5,0,7,6,7]
k = 4
topk_sort_left(arr, k)
```

只排序后k个大的数

```python
#只排序后k个大的数
#获得前n-k小的数O(n)，进行快排O(klogk)
def topk_sort_right(nums, k):
    topk_split(nums, len(nums)-k, 0, len(nums)-1) 
    topk = nums[len(nums)-k:]
    quicksort(topk, 0, len(topk)-1)
    return nums[:len(nums)-k]+topk #只排序后k个数字

arr = [0,0,1,3,4,5,0,-7,6,7]
k = 4
print(topk_sort_right(arr, k))
```







### Bit

#### 191. Number of 1 Bits

```python
class Solution:
    def hammingWeight(self, n: int) -> int:
        ans = 0
        while n != 0:
            n &= n - 1
            ans += 1
        return ans
```

#### 338. Counting Bits

```python
class Solution:
    def countBits(self, n: int) -> List[int]:
        def bitcount(x):
            count = 0
            while x != 0:
                x &= x - 1
                count += 1
            return count
        
        ans = []
        for i in range(n + 1):
            ans.append(bitcount(i))
        
        return ans
```



#### 89. Gray Code

```python
class Solution:
    def grayCode(self, n: int) -> List[int]:
        res, head = [0], 1
        for i in range(n):
            for j in range(len(res) - 1, -1, -1):
                res.append(head + res[j])
                print(head, res[j])
            head <<= 1
        return res
```

### Prefix

https://leetcode-cn.com/problems/subarray-sum-equals-k/solution/de-liao-yi-wen-jiang-qian-zhui-he-an-pai-yhyf/

#### 724

#### 1

#### 560. Subarray Sum Equals K

```python
class Solution:
    def subarraySum(self, nums: List[int], k: int) -> int:
        
        p = [0]
        for i in nums: p.append(p[-1] + i)
        count = collections.Counter()
        # count[0] = 1 # 这一步不需要，因为初始化 p = [0] 包含了后面i从0开始,必定有count[0]+=1
        
        ans = 0
        for i in p:
            if i - k in count:
                ans += count[i - k]
            count[i] += 1
        
        return ans
```

另外一种简便的写法，本质都是 j - i = k 等价于 i = j - k or j = i + k

```python
class Solution:
    def subarraySum(self, nums: List[int], k: int) -> int:

        p = [0]
        for i in nums: p.append(p[-1] + i)
        count = collections.Counter()
        
        ans = 0
        for i in p:
            ans += count[i]
            count[i + k] += 1
            
        return ans
```



#### 1248

#### 974

#### 523

#### 930. Binary Subarrays With Sum

```python
class Solution:
    def numSubarraysWithSum(self, nums: List[int], goal: int) -> int:
        p = [0]
        for i in nums: p.append(p[-1] + i)
        count = collections.Counter()
        
        ans = 0
        for j in p:
            ans += count[j]
            count[j + goal] += 1
            
        return ans
```



#### 

### Others

#### 7. Reverse Integer

Given a signed 32-bit integer `x`, return `x` *with its digits reversed*. If reversing `x` causes the value to go outside the signed 32-bit integer range `[-231, 231 - 1]`, then return `0`.

**Example 1:**

```
Input: x = 123
Output: 321
```

```python
def reverse(self, x: int) -> int:
    sigh = 1
    if x < 0:
        sigh = -1
        x = -x
        rev = int(str(x)[::-1])
    return 0 if rev > pow(2, 31) else rev * sigh
```

```python
def reverse(self, x: int) -> int:    
    sigh = 1
    if x < 0:
        sigh = -1
        x = -x
    rev = 0
    while x:
        rev = rev * 10 + x % 10
        x //= 10
    return 0 if rev > pow(2, 31) else rev * sigh
```



#### 48. Rotate Image

You are given an `n x n` 2D `matrix` representing an image, rotate the image by **90** degrees (clockwise).

You have to rotate the image [**in-place**](https://en.wikipedia.org/wiki/In-place_algorithm), which means you have to modify the input 2D matrix directly. **DO NOT** allocate another 2D matrix and do the rotation.

 

**Example 1:**

![img](https://assets.leetcode.com/uploads/2020/08/28/mat1.jpg)

```
Input: matrix = [[1,2,3],[4,5,6],[7,8,9]]
Output: [[7,4,1],[8,5,2],[9,6,3]]
```

```python
# Approach 1: Rotate Groups of Four Cells
class Solution:
    def rotate(self, matrix: List[List[int]]) -> None:
        n = len(matrix[0])
        for i in range(n // 2 + n % 2):
            for j in range(n // 2):
                tmp = matrix[n - 1 - j][i]
                matrix[n - 1 - j][i] = matrix[n - 1 - i][n - j - 1]
                matrix[n - 1 - i][n - j - 1] = matrix[j][n - 1 - i]
                matrix[j][n - 1 - i] = matrix[i][j]
                matrix[i][j] = tmp
```

```python
# Approach 2: Transpose and then Reverse Left to Right
class Solution:
    def rotate(self, matrix: List[List[int]]) -> None:
        self.transpose(matrix)
        self.reflect(matrix)
    
    def transpose(self, matrix):
        n = len(matrix)
        for i in range(n):
            for j in range(i, n):
                matrix[j][i], matrix[i][j] = matrix[i][j], matrix[j][i]

    def reflect(self, matrix):
        n = len(matrix)
        for i in range(n):
            for j in range(n // 2):
                matrix[i][j], matrix[i][-j - 1] = matrix[i][-j - 1], matrix[i][j]

```





#### 58. Length of Last Word

Given a string `s` consists of some words separated by spaces, return *the length of the last word in the string. If the last word does not exist, return* `0`.

A **word** is a maximal substring consisting of non-space characters only.

 

**Example 1:**

```
Input: s = "Hello World"
Output: 5
```

```python
    def lengthOfLastWord(self, s: str) -> int:
        length = 0
        for i in range(len(s)-1,-1,-1):
            if s[i] != ' ':
                length += 1
            elif length > 0:
                break
        return length
```



#### 73. Set Matrix Zeroes

Given an `*m* x *n*` matrix. If an element is **0**, set its entire row and column to **0**. Do it [**in-place**](https://en.wikipedia.org/wiki/In-place_algorithm).

**Follow up:**

- A straight forward solution using O(*m**n*) space is probably a bad idea.
- A simple improvement uses O(*m* + *n*) space, but still not the best solution.
- Could you devise a constant space solution?

 

**Example 1:**

![img](https://assets.leetcode.com/uploads/2020/08/17/mat1.jpg)

```
Input: matrix = [[1,1,1],[1,0,1],[1,1,1]]
Output: [[1,0,1],[0,0,0],[1,0,1]]
```

**Example 2:**

![img](https://assets.leetcode.com/uploads/2020/08/17/mat2.jpg)

```
Input: matrix = [[0,1,2,0],[3,4,5,2],[1,3,1,5]]
Output: [[0,0,0,0],[0,4,5,0],[0,3,1,0]]
```

```python
class Solution:
    def setZeroes(self, matrix: List[List[int]]) -> None:
        """
        Do not return anything, modify matrix in-place instead.
        """
        m = len(matrix)
        n = len(matrix[0])
        rows, cols = set(), set()
        for i in range(m):
            for j in range(n):
                if matrix[i][j] == 0:
                    rows.add(i)
                    cols.add(j)
        for i in range(m):
            for j in range(n):
                if i in rows or j in cols:
                    matrix[i][j] = 0
        
```

```python
class Solution:
    def setZeroes(self, matrix: List[List[int]]) -> None:
        """
        Do not return anything, modify matrix in-place instead.
        """
        m = len(matrix)
        n = len(matrix[0])
        
        first_row_zero = False
        first_col_zero = False
        
        for j in range(n):
            if matrix[0][j] == 0: first_row_zero = True
        
        for i in range(m):
            if matrix[i][0] == 0: first_col_zero = True
            
        for i in range(1, m):
            for j in range(1, n):
                if matrix[i][j] == 0:
                    matrix[0][j] = 0
                    matrix[i][0] = 0
        for i in range(1, m):
            for j in range(1, n):
                if not matrix[0][j] or not matrix[i][0]:
                    matrix[i][j] = 0
        
        if first_row_zero:
            for j in range(n):
                matrix[0][j] = 0
                
        if first_col_zero:
            for i in range(m):
                matrix[i][0] = 0
```



#### 66. Plus One

Given a **non-empty** array of decimal digits representing a non-negative integer, increment one to the integer.

The digits are stored such that the most significant digit is at the head of the list, and each element in the array contains a single digit.

You may assume the integer does not contain any leading zero, except the number 0 itself.

**Example 1:**

```
Input: digits = [1,2,3]
Output: [1,2,4]
Explanation: The array represents the integer 123.
```

```python
class Solution:
    def plusOne(self, digits: List[int]) -> List[int]:
        n = len(digits)

        # move along the input array starting from the end
        for i in range(n):
            idx = n - 1 - i
            # set all the nines at the end of array to zeros
            if digits[idx] == 9:
                digits[idx] = 0
            # here we have the rightmost not-nine
            else:
                # increase this rightmost not-nine by 1
                digits[idx] += 1
                # and the job is done
                return digits

        # we're here because all the digits are nines
        return [1] + digits
```

```python
class Solution:
    def plusOne(self, digits: List[int]) -> List[int]:
        num = 0
        for i in range(len(digits)-1,-1,-1):
            num += digits[i] * 10 ** (len(digits) - i - 1)
        return [i for i in str(num+1)]
```



#### 118. Pascal's Triangle

Given an integer `numRows`, return the first numRows of **Pascal's triangle**.

In **Pascal's triangle**, each number is the sum of the two numbers directly above it as shown:

![img](https://upload.wikimedia.org/wikipedia/commons/0/0d/PascalTriangleAnimated2.gif)

**Example 1:**

```
Input: numRows = 5
Output: [[1],[1,1],[1,2,1],[1,3,3,1],[1,4,6,4,1]]
```

```python
def generate(self, numRows: int) -> List[List[int]]:
    result = []
    for i in range(numRows):
        result.append([])
        for j in range(i+1):
            if j in (0, i):
                result[i].append(1)
            else:
                result[i].append(result[i-1][j-1] + result[i-1][j])
    return result
```



#### 119. Pascal's Triangle II

Given an integer `rowIndex`, return the `rowIndexth` (**0-indexed**) row of the **Pascal's triangle**.

In **Pascal's triangle**, each number is the sum of the two numbers directly above it as shown:

![img](https://upload.wikimedia.org/wikipedia/commons/0/0d/PascalTriangleAnimated2.gif)

```python
def getRow(self, rowIndex: int) -> List[int]:
    row = [1] * (rowIndex + 1)
    for i in range(1, rowIndex + 1):
        row[i] = int(row[i-1] * (rowIndex - i + 1)/i)
    return row
```



```python
def getRow(self, rowIndex: int) -> List[int]:
    ret = [1] * (rowIndex + 1)
    for i in range(2, rowIndex + 1):
        for j in range(i - 1, 0, -1):
            ret[j] += ret[j - 1]
    return ret
```



```python
def getRow(self, rowIndex: int) -> List[int]: # dp 
    ret = []
    for i in range(1, rowIndex + 2):
        tmp = [1 for _ in range(i)]
        for j in range(1, len(tmp) - 1):
            # 这里注意是 i - 2
            tmp[j] = ret[i - 2][j - 1] + ret[i - 2][j]
        ret.append(tmp)
    return ret[-1]
```



#### 128. Longest Consecutive Sequence

Given an unsorted array of integers `nums`, return *the length of the longest consecutive elements sequence.*

You must write an algorithm that runs in `O(n)` time.

 

**Example 1:**

```
Input: nums = [100,4,200,1,3,2]
Output: 4
Explanation: The longest consecutive elements sequence is [1, 2, 3, 4]. Therefore its length is 4.
```

```python
class Solution:
    def longestConsecutive(self, nums: List[int]) -> int:
        hash_map = set(nums)
        trueLen = 0
        for num in nums:
            if num - 1 not in hash_map:
                currNum = num
                tempLen = 1
                
                while currNum + 1 in hash_map:
                    currNum += 1
                    tempLen += 1
                
                trueLen = max(trueLen, tempLen)
                
        return trueLen
```

```python
class Solution:
    def longestConsecutive(self, nums: List[int]) -> int:
        if len(nums) <= 1:
            return len(nums)
        
        nums.sort()
        ans = temp = 1
        for i in range(1, len(nums)):
            if nums[i] == nums[i-1]:
                continue
            elif nums[i] == nums[i-1] + 1:
                temp += 1
            else:
                temp = 1
            ans = max(ans, temp)
            
        return ans
```





#### 136*. Single Number

Given a **non-empty** array of integers `nums`, every element appears *twice* except for one. Find that single one.

You must implement a solution with a linear runtime complexity and use only constant extra space.

**Example 1:**

```
Input: nums = [2,2,1]
Output: 1
```

```python
def singleNumber(self, nums: List[int]) -> int:
    a = 0
    for i in nums:
        a ^= i
    return a
```

```python
def singleNumber(self, nums):
    return 2 * sum(set(nums)) - sum(nums)
```

```python
from collections import defaultdict
class Solution:
    def singleNumber(self, nums: List[int]) -> int:
        hash_table = defaultdict(int)
        for i in nums:
            hash_table[i] += 1
        
        for i in hash_table:
            if hash_table[i] == 1:
                return i
```

#### 165. Compare Version Numbers

Given two version numbers, `version1` and `version2`, compare them.

**Example 1:**

```
Input: version1 = "1.01", version2 = "1.001"
Output: 0
Explanation: Ignoring leading zeroes, both "01" and "001" represent the same integer "1".
```

**Example 2:**

```
Input: version1 = "1.0", version2 = "1.0.0"
Output: 0
Explanation: version1 does not specify revision 2, which means it is treated as "0".
```

**Example 3:**

```
Input: version1 = "0.1", version2 = "1.1"
Output: -1
Explanation: version1's revision 0 is "0", while version2's revision 0 is "1". 0 < 1, so version1 < version2.
```

```python
class Solution:
    def compareVersion(self, version1: str, version2: str) -> int:
        version1 = version1.split('.')
        version2 = version2.split('.')
        
        n = max(len(version1), len(version2))
        
        for i in range(n):
            v1 = int(version1[i]) if i < len(version1) else 0
            v2 = int(version2[i]) if i < len(version2) else 0
            
            if v1 > v2:
                return 1
            elif v1 < v2:
                return -1
            
        return 0
```





#### 169. ! Majority Element

Given an array `nums` of size `n`, return *the majority element*.

The majority element is the element that appears more than `⌊n / 2⌋` times. You may assume that the majority element always exists in the array.

**Example 1:**

```
Input: nums = [3,2,3]
Output: 3
```

```python
class Solution:
    def majorityElement(self, nums):
        count = 0
        for num in nums:
            if count == 0: candidate = num
            count += (1 if num == candidate else -1)
        return candidate
```





#### 189. Rotate Array

Given an array, rotate the array to the right by `k` steps, where `k` is non-negative.

 

**Example 1:**

```
Input: nums = [1,2,3,4,5,6,7], k = 3
Output: [5,6,7,1,2,3,4]
Explanation:
rotate 1 steps to the right: [7,1,2,3,4,5,6]
rotate 2 steps to the right: [6,7,1,2,3,4,5]
rotate 3 steps to the right: [5,6,7,1,2,3,4]
```

```python
# Approach 1:Brute Force
class Solution:
    def rotate(self, nums: List[int], k: int) -> None:
        # speed up the rotation
        k %= len(nums)

        for i in range(k):
            previous = nums[-1]
            for j in range(len(nums)):
                nums[j], previous = previous, nums[j]
```

```python
# Approach 4:Using Reverse
class Solution:
    def reverse(self, nums: list, start: int, end: int) -> None:
        while start < end:
            nums[start], nums[end] = nums[end], nums[start]
            start, end = start + 1, end - 1
                
    def rotate(self, nums: List[int], k: int) -> None:
        n = len(nums)
        k %= n

        self.reverse(nums, 0, n - 1)
        self.reverse(nums, 0, k - 1)
        self.reverse(nums, k, n - 1)
```

#### 231. Power of Two

Given an integer `n`, return *`true` if it is a power of two. Otherwise, return `false`*.

An integer `n` is a power of two, if there exists an integer `x` such that `n == 2x`.

 

**Example 1:**

```
Input: n = 1
Output: true
Explanation: 20 = 1
```

```python
class Solution:
    BIG = 2**30
    def isPowerOfTwo(self, n: int) -> bool:
        return n > 0 and Solution.BIG % n == 0
```

```python
class Solution:
    def isPowerOfTwo(self, n: int) -> bool:
        return n > 0 and (n & (n - 1)) == 0
```

```python
class Solution:
    def isPowerOfTwo(self, n: int) -> bool:
        return n > 0 and (n & -n) == n
```

### Brainteasers

#### 621. Task Scheduler

```python
class Solution:
    def leastInterval(self, tasks: List[str], n: int) -> int:
        dict_count = list(collections.Counter(tasks).values())
        M = max(dict_count)
        N = len(tasks)
        K = dict_count.count(M)
        return max(N, (M - 1)*(n + 1) + K)
```



------

### After 300

| Problem                                     | Level  | Type          | Comment                                                      |
| :------------------------------------------ | :----- | ------------- | ------------------------------------------------------------ |
| 202. Happy Numbers                          | Easy   | Dict          | Tricky Dict; Squares by digits                               |
| 203. Remove Linked List Elements            | Easy   | List          |                                                              |
| 242. Valid Anagram                          | Easy   | Dict          | Standard Dict                                                |
| 332. Reconstruct Itinerary                  | Medium | DFS           | Tricky DFS: need dict as Iterator; how to sort and prevent loop |
| 349. Intersection of Two Arrays             | Easy   | Dict          |                                                              |
| 367. Valid Perfect Square                   | Easy   | Binary Search | Standard BS                                                  |
| 454. 4Sum II                                | Medium | Dict          | Tricky Dict                                                  |
| 459*. Repeated Substring Pattern            | Easy   | String        | Tricky string problem                                        |
| 491. Increasing Subsequences                | Medium | DFS           | use "used" to same level Pruning                             |
| 977. Squares of a Sorted Array              | Easy   | Two Pointer   |                                                              |
| 1047. Remove All Adjacent Duplicates        | Easy   | Stack         | Standard Stack; same as 20                                   |
| 1306. Jump Game III                         | Medium | BFS           | Standard BFS                                                 |
| 1897. Redistribute Characters               | Medium | Dict          | Standard Dict                                                |
| 1898. Maximum Number of Removable           | Medium | Binary Search | subSeq func; hard to realize should use Binary Search        |
| 1899. Merge Triplets to Form Target Triplet | Medium | Array         | Tricky array problem                                         |



------

#### 202. Happy Number

Write an algorithm to determine if a number `n` is happy.

A **happy number** is a number defined by the following process:

- Starting with any positive integer, replace the number by the sum of the squares of its digits.
- Repeat the process until the number equals 1 (where it will stay), or it **loops endlessly in a cycle** which does not include 1.
- Those numbers for which this process **ends in 1** are happy.

Return `true` *if* `n` *is a happy number, and* `false` *if not*.

 

**Example 1:**

```
Input: n = 19
Output: true
Explanation:
12 + 92 = 82
82 + 22 = 68
62 + 82 = 100
12 + 02 + 02 = 1
```

```python
class Solution:
    def isHappy(self, n: int) -> bool:
        seen = set()
        
        while n != 1:
            n = sum([int(x)**2 for x in str(n)])
            if n in seen:
                return False
            else:
                seen.add(n)
        return True
```

```python
# 按位求平方和的其他办法
    def getSum(n):
        sum_ = 0
        while n > 0:
            sum_ += (n%10) * (n%10)
            n //= 10
        return sum_
    
    def get_next(n):
        sum_ = 0
        while n > 0:
            n, digit = divmod(n, 10)
            sum_ += digit ** 2
        return sum_

```







#### 203. Remove Linked List Elements

Given the `head` of a linked list and an integer `val`, remove all the nodes of the linked list that has `Node.val == val`, and return *the new head*.

 

**Example 1:**

![img](https://assets.leetcode.com/uploads/2021/03/06/removelinked-list.jpg)

```
Input: head = [1,2,6,3,4,5,6], val = 6
Output: [1,2,3,4,5]
```

```python
class Solution:
    def removeElements(self, head, val):

        dummy_head = ListNode(-1, head)
        
        curr = dummy_head
        while curr.next != None:
            if curr.next.val == val:
                curr.next = curr.next.next
            else:
                curr = curr.next
                
        return dummy_head.next
```



#### 225. Implement Stack using Queues

Implement a last in first out (LIFO) stack using only two queues. The implemented stack should support all the functions of a normal queue (`push`, `top`, `pop`, and `empty`).

```python
class MyStack:

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.queue = collections.deque()


    def push(self, x: int) -> None:
        """
        Push element x onto stack.
        """
        n = len(self.queue)
        self.queue.append(x)
        for _ in range(n):
            self.queue.append(self.queue.popleft())


    def pop(self) -> int:
        """
        Removes the element on top of the stack and returns that element.
        """
        return self.queue.popleft()


    def top(self) -> int:
        """
        Get the top element.
        """
        return self.queue[0]


    def empty(self) -> bool:
        """
        Returns whether the stack is empty.
        """
        return not self.queue
```



#### 232. Implement Queue using Stacks

Implement a first in first out (FIFO) queue using only two stacks. The implemented queue should support all the functions of a normal queue (`push`, `peek`, `pop`, and `empty`).

```python
# 使用两个栈实现先进先出的队列
class MyQueue:
    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.stack1 = list()  # 输入栈
        self.stack2 = list()  # 输出栈

    def push(self, x: int) -> None:
        """
        Push element x to the back of queue.
        """
        # self.stack1用于接受元素
        self.stack1.append(x)

    def pop(self) -> int:
        """
        Removes the element from in front of queue and returns that element.
        """
        # self.stack2用于弹出元素，如果self.stack2为[],则将self.stack1中元素全部弹出给self.stack2
        if self.stack2 == []:
            while self.stack1:
                tmp = self.stack1.pop()
                self.stack2.append(tmp)
        return self.stack2.pop()

    def peek(self) -> int:
        """
        Get the front element.
        """
        if self.stack2 == []:
            while self.stack1:
                tmp = self.stack1.pop()
                self.stack2.append(tmp)
        return self.stack2[-1]

    def empty(self) -> bool:
        """
        Returns whether the queue is empty.
        """
        return self.stack1 == [] and self.stack2 == []

```

#### 238. Product of Array Except Self

```python
class Solution:
    def productExceptSelf(self, nums: List[int]) -> List[int]:
        n = len(nums)
        ans = [1] * n 
        for i in range(1, n):
            ans[i] = ans[i-1] * nums[i-1]
            
        R = 1
        for i in reversed(range(n)):
            ans[i] *= R
            R *= nums[i]
        return ans
```





#### 240. Search a 2D Matrix II

Write an efficient algorithm that searches for a `target` value in an `m x n` integer `matrix`. The `matrix` has the following properties:

- Integers in each row are sorted in ascending from left to right.
- Integers in each column are sorted in ascending from top to bottom.

**Example 1:**

![img](https://assets.leetcode.com/uploads/2020/11/24/searchgrid2.jpg)

```
Input: matrix = [[1,4,7,11,15],[2,5,8,12,19],[3,6,9,16,22],[10,13,14,17,24],[18,21,23,26,30]], target = 5
Output: true
```

```python
# bisect
class Solution:
    def binary_search(self, matrix, target, start, vertical):
        lo = start
        hi = len(matrix[0])-1 if vertical else len(matrix)-1

        while hi >= lo:
            mid = (lo + hi)//2
            if vertical: # searching a column
                if matrix[start][mid] < target:
                    lo = mid + 1
                elif matrix[start][mid] > target:
                    hi = mid - 1
                else:
                    return True
            else: # searching a row
                if matrix[mid][start] < target:
                    lo = mid + 1
                elif matrix[mid][start] > target:
                    hi = mid - 1
                else:
                    return True
        
        return False

    def searchMatrix(self, matrix, target):
        # an empty matrix obviously does not contain `target`
        if not matrix:
            return False

        # iterate over matrix diagonals starting in bottom left.
        for i in range(min(len(matrix), len(matrix[0]))):
            vertical_found = self.binary_search(matrix, target, i, True)
            horizontal_found = self.binary_search(matrix, target, i, False)
            if vertical_found or horizontal_found:
                return True
        
        return False
```

```python
# move from left bottom
class Solution:
    def searchMatrix(self, matrix, target):
        # an empty matrix obviously does not contain `target` (make this check
        # because we want to cache `width` for efficiency's sake)
        if len(matrix) == 0 or len(matrix[0]) == 0:
            return False

        # cache these, as they won't change.
        height = len(matrix)
        width = len(matrix[0])

        # start our "pointer" in the bottom-left
        row = height-1
        col = 0

        while col < width and row >= 0:
            if matrix[row][col] > target:
                row -= 1
            elif matrix[row][col] < target:
                col += 1
            else: # found it
                return True
        
        return False
```



#### 242. Valid Anagram

Given two strings `s` and `t`, return `true` *if* `t` *is an anagram of* `s`*, and* `false` *otherwise*.

 

**Example 1:**

```
Input: s = "anagram", t = "nagaram"
Output: true
```

```python
class Solution:
    def isAnagram(self, s: str, t: str) -> bool:
        Dict = defaultdict(int)
        for i in s:
            Dict[i] += 1
        for i in t:
            Dict[i] -= 1
        for i in Dict.values():
            if i != 0:
                return False
        return True
```





#### 343. Integer Break

Given an integer `n`, break it into the sum of `k` **positive integers**, where `k >= 2`, and maximize the product of those integers.

Return *the maximum product you can get*.

 

**Example 1:**

```
Input: n = 2
Output: 1
Explanation: 2 = 1 + 1, 1 × 1 = 1.
```

**Example 2:**

```
Input: n = 10
Output: 36
Explanation: 10 = 3 + 3 + 4, 3 × 3 × 4 = 36.
```

```python
# DP
class Solution:
    def integerBreak(self, n: int) -> int:
        dp = [0] * (n + 1)
        dp[2] = 1
        for i in range(3, n + 1):
            # 假设对正整数 i 拆分出的第一个正整数是 j（1 <= j < i），则有以下两种方案：
            # 1) 将 i 拆分成 j 和 i−j 的和，且 i−j 不再拆分成多个正整数，此时的乘积是 j * (i-j)
            # 2) 将 i 拆分成 j 和 i−j 的和，且 i−j 继续拆分成多个正整数，此时的乘积是 j * dp[i-j]
            for j in range(1, i - 1):
                dp[i] = max(dp[i], max(j * (i - j), j * dp[i - j]))
        return dp[n]
```

```python
# 优化DP
class Solution:
    def integerBreak(self, n: int) -> int:
        if n < 4:
            return n - 1
        dp = [0] * (n + 1)
        dp[2] = 1
        for i in range(3, n + 1):
            dp[i] = max(2 * (i - 2), 2 * dp[i - 2], 3 * (i - 3), 3 * dp[i - 3])
        
        return dp[n]
```

```python
# math
class Solution:
    def integerBreak(self, n: int) -> int:
        if n <= 3:
            return n - 1
        
        quotient, remainder = n // 3, n % 3
        if remainder == 0:
            return 3 ** quotient
        elif remainder == 1:
            return 3 ** (quotient - 1) * 4
        else:
            return 3 ** quotient * 2
```



#### 347*. Top K Frequent Elements

Given an integer array `nums` and an integer `k`, return *the* `k` *most frequent elements*. You may return the answer in **any order**.

 

**Example 1:**

```
Input: nums = [1,1,1,2,2,3], k = 2
Output: [1,2]
```

```python
from collections import Counter
class Solution:
    def topKFrequent(self, nums: List[int], k: int) -> List[int]: 
        # O(1) time 
        if k == len(nums):
            return nums
        
        # 1. build hash map : character and how often it appears
        # O(N) time
        count = Counter(nums)   
        # 2-3. build heap of top k frequent elements and
        # convert it into an output array
        # O(N log k) time
        return heapq.nlargest(k, count.keys(), key=count.get)
      # return sorted(count, key=count.get, reverse=True)[:k]
```

```python
#时间复杂度：O(nlogk)
#空间复杂度：O(n)
import heapq
class Solution:
    def topKFrequent(self, nums: List[int], k: int) -> List[int]:
        #要统计元素出现频率
        map_ = {} #nums[i]:对应出现的次数
        for i in range(len(nums)):
            map_[nums[i]] = map_.get(nums[i], 0) + 1
        
        #对频率排序
        #定义一个小顶堆，大小为k
        pri_que = [] #小顶堆
        
        #用固定大小为k的小顶堆，扫面所有频率的数值
        for key, freq in map_.items():
            heapq.heappush(pri_que, (freq, key))
            if len(pri_que) > k: #如果堆的大小大于了K，则队列弹出，保证堆的大小一直为k
                heapq.heappop(pri_que)
        
        #找出前K个高频元素，因为小顶堆先弹出的是最小的，所以倒叙来输出到数组
        result = [0] * k
        for i in range(k-1, -1, -1):
            result[i] = heapq.heappop(pri_que)[1]
        return result
```





#### 349. Intersection of Two Arrays





#### 376. Wiggle Subsequence

A **wiggle sequence** is a sequence where the differences between successive numbers strictly alternate between positive and negative. The first difference (if one exists) may be either positive or negative. A sequence with one element and a sequence with two non-equal elements are trivially wiggle sequences.

- For example, `[1, 7, 4, 9, 2, 5]` is a **wiggle sequence** because the differences `(6, -3, 5, -7, 3)` alternate between positive and negative.
- In contrast, `[1, 4, 7, 2, 5]` and `[1, 7, 4, 5, 5]` are not wiggle sequences. The first is not because its first two differences are positive, and the second is not because its last difference is zero.

A **subsequence** is obtained by deleting some elements (possibly zero) from the original sequence, leaving the remaining elements in their original order.

Given an integer array `nums`, return *the length of the longest **wiggle subsequence** of* `nums`.

 

**Example 1:**

```
Input: nums = [1,7,4,9,2,5]
Output: 6
Explanation: The entire sequence is a wiggle sequence with differences (6, -3, 5, -7, 3).
```

```python
class Solution:
    def wiggleMaxLength(self, nums: List[int]) -> int:
        preC,curC,res = 0,0,1  #题目里nums长度大于等于1，当长度为1时，其实到不了for循环里去，所以不用考虑nums长度
        for i in range(len(nums) - 1):
            curC = nums[i + 1] - nums[i]
            if curC * preC <= 0 and curC !=0:  #差值为0时，不算摆动
                res += 1
                preC = curC  #如果当前差值和上一个差值为一正一负时，才需要用当前差值替代上一个差值
        return res
```







#### 383. Ransom Note

Given two stings `ransomNote` and `magazine`, return `true` if `ransomNote` can be constructed from `magazine` and `false` otherwise.

Each letter in `magazine` can only be used once in `ransomNote`.

 

**Example 1:**

```
Input: ransomNote = "a", magazine = "b"
Output: false
```

```python
class Solution:
    def canConstruct(self, ransomNote: str, magazine: str) -> bool:
        x=Counter(ransomNote)
        y=Counter(magazine)
        for i,v in x.items():
            if(x[i]<=y[i]):
                continue
            else:
                return False
        return True
```



#### 406. Queue Reconstruction by Height

You are given an array of people, `people`, which are the attributes of some people in a queue (not necessarily in order). Each `people[i] = [hi, ki]` represents the `ith` person of height `hi` with **exactly** `ki` other people in front who have a height greater than or equal to `hi`.

Reconstruct and return *the queue that is represented by the input array* `people`. The returned queue should be formatted as an array `queue`, where `queue[j] = [hj, kj]` is the attributes of the `jth` person in the queue (`queue[0]` is the person at the front of the queue).

 

**Example 1:**

```
Input: people = [[7,0],[4,4],[7,1],[5,0],[6,1],[5,2]]
Output: [[5,0],[7,0],[5,2],[6,1],[4,4],[7,1]]
Explanation:
Person 0 has height 5 with no other people taller or the same height in front.
Person 1 has height 7 with no other people taller or the same height in front.
Person 2 has height 5 with two persons taller or the same height in front, which is person 0 and 1.
Person 3 has height 6 with one person taller or the same height in front, which is person 1.
Person 4 has height 4 with four people taller or the same height in front, which are people 0, 1, 2, and 3.
Person 5 has height 7 with one person taller or the same height in front, which is person 1.
Hence [[5,0],[7,0],[5,2],[6,1],[4,4],[7,1]] is the reconstructed queue.
```

```python
class Solution:
    def reconstructQueue(self, people: List[List[int]]) -> List[List[int]]:
        people.sort(key=lambda x: (-x[0], x[1])) # first sort -x[0], if equal, sort x[1]
        que = []
        for p in people:
            if p[1] > len(que):
                que.append(p)
            else:
                que.insert(p[1], p) # que[p[1]:p[1]] = [p]
        return que
```



#### 435. Non-overlapping Intervals

Given an array of intervals `intervals` where `intervals[i] = [starti, endi]`, return *the minimum number of intervals you need to remove to make the rest of the intervals non-overlapping*.

 

**Example 1:**

```
Input: intervals = [[1,2],[2,3],[3,4],[1,3]]
Output: 1
Explanation: [1,3] can be removed and the rest of the intervals are non-overlapping.
```

```python
class Solution:
    def eraseOverlapIntervals(self, intervals: List[List[int]]) -> int:
        if not intervals:
            return 0
        intervals.sort(key=lambda x: x[1])
        pos = intervals[0][1]
        ans = 1
        for interval in intervals:
            if interval[0] >= pos:
                pos = interval[1]
                ans += 1
        return len(intervals) - ans
```





#### 452. Minimum Number of Arrows to Burst Balloons

There are some spherical balloons spread in two-dimensional space. For each balloon, provided input is the start and end coordinates of the horizontal diameter. Since it's horizontal, y-coordinates don't matter, and hence the x-coordinates of start and end of the diameter suffice. The start is always smaller than the end.

An arrow can be shot up exactly vertically from different points along the x-axis. A balloon with `xstart` and `xend` bursts by an arrow shot at `x` if `xstart ≤ x ≤ xend`. There is no limit to the number of arrows that can be shot. An arrow once shot keeps traveling up infinitely.

Given an array `points` where `points[i] = [xstart, xend]`, return *the minimum number of arrows that must be shot to burst all balloons*.

 

**Example 1:**

```
Input: points = [[10,16],[2,8],[1,6],[7,12]]
Output: 2
Explanation: One way is to shoot one arrow for example at x = 6 (bursting the balloons [2,8] and [1,6]) and another arrow at x = 11 (bursting the other two balloons).
```

```python
# 求交集
class Solution:
    def findMinArrowShots(self, points: List[List[int]]) -> int:
        points.sort()
        i = 1
        while i < len(points):
            (al, ar), (bl, br) = points[i - 1], points[i]
            if bl <= ar:
                points[i - 1] = bl, min(ar, br)
                points.pop(i)
            else:
                i += 1
        return len(points)
```

```python
# 按第二位排序
# 箭永远射气球最右端
class Solution:
    def findMinArrowShots(self, points: List[List[int]]) -> int:
        if not points:
            return 0
        
        points.sort(key=lambda balloon: balloon[1])
        pos = points[0][1]
        ans = 1
        for balloon in points:
            if balloon[0] > pos:
                pos = balloon[1]
                ans += 1
        
        return ans
```

```python
# 按第一位排序
# 箭永远射重叠气球最小右边界
class Solution:
    def findMinArrowShots(self, points: List[List[int]]) -> int:
        if len(points) == 0: return 0
        points.sort(key=lambda x: x[0])
        result = 1
        pos = points[0][1] # 起始射击位置为气球最右端
        for i in range(1, len(points)):
            if points[i][0] > pos: 
                result += 1
                pos = points[i][1] # 更新射击位置为新的不重叠气球的最右端
            else:
                pos = min(pos, points[i][1]) # 更新重叠气球最小右边界
        return result
```





#### 454. 4Sum II

Given four integer arrays `nums1`, `nums2`, `nums3`, and `nums4` all of length `n`, return the number of tuples `(i, j, k, l)` such that:

- `0 <= i, j, k, l < n`
- `nums1[i] + nums2[j] + nums3[k] + nums4[l] == 0`

 

**Example 1:**

```
Input: nums1 = [1,2], nums2 = [-2,-1], nums3 = [-1,2], nums4 = [0,2]
Output: 2
Explanation:
The two tuples are:
1. (0, 0, 0, 1) -> nums1[0] + nums2[0] + nums3[0] + nums4[1] = 1 + (-2) + (-1) + 2 = 0
2. (1, 1, 0, 0) -> nums1[1] + nums2[1] + nums3[0] + nums4[0] = 2 + (-1) + (-1) + 0 = 0
```

```python
class Solution:
    def fourSumCount(self, nums1: List[int], nums2: List[int], nums3: List[int], nums4: List[int]) -> int:
        
        hash_map = defaultdict(int)
        for i in nums1:
            for j in nums2:
                hash_map[i + j] += 1
        count = 0  
        for i in nums3:
            for j in nums4:
                key = -i-j
                if key in hash_map.keys():
                    count += hash_map[key]
                    
        return count
```



#### 455. Assign Cookies

Assume you are an awesome parent and want to give your children some cookies. But, you should give each child at most one cookie.

Each child `i` has a greed factor `g[i]`, which is the minimum size of a cookie that the child will be content with; and each cookie `j` has a size `s[j]`. If `s[j] >= g[i]`, we can assign the cookie `j` to the child `i`, and the child `i` will be content. Your goal is to maximize the number of your content children and output the maximum number.

 

**Example 1:**

```
Input: g = [1,2,3], s = [1,1]
Output: 1
Explanation: You have 3 children and 2 cookies. The greed factors of 3 children are 1, 2, 3. 
And even though you have 2 cookies, since their size is both 1, you could only make the child whose greed factor is 1 content.
You need to output 1.
```

```python
class Solution:
    def findContentChildren(self, g: List[int], s: List[int]) -> int:
        g.sort()
        s.sort()
        res = 0
        for i in range(len(s)):
            if res <len(g) and s[i] >= g[res]:  #小饼干先喂饱小胃口
                res += 1
        return res
```







#### 459*. Repeated Substring Pattern

Given a string `s`, check if it can be constructed by taking a substring of it and appending multiple copies of the substring together.

 

**Example 1:**

```
Input: s = "abab"
Output: true
Explanation: It is the substring "ab" twice.
```

```python
# copy original string
class Solution:
    def repeatedSubstringPattern(self, s: str) -> bool:
        return (s + s).find(s, 1) != len(s)
```

```python
# enumerate
class Solution:
    def repeatedSubstringPattern(self, s: str) -> bool:
        n = len(s)
        for i in range(1, n // 2 + 1):
            if n % i == 0:
                if all(s[j] == s[j - i] for j in range(i, n)):
                    return True
        return False
```

```python
# KMP
class Solution:
    def repeatedSubstringPattern(self, s: str) -> bool:  
        if len(s) == 0:
            return False
        nxt = self.getNext(s)
        if nxt[-1] != 0 and len(s) % (len(s) - nxt[-1]) == 0:
            return True
        return False

    def getNext(self, s):
        next = [0] * len(s)
        j = 0
        for i in range(1, len(s)):
            while (j > 0 and s[i] != s[j]):
                j = next[j-1]
            if s[i] == s[j]:
                j += 1
            next[i] = j
        return next
```









#### 509. Fibonacci Number

The **Fibonacci numbers**, commonly denoted `F(n)` form a sequence, called the **Fibonacci sequence**, such that each number is the sum of the two preceding ones, starting from `0` and `1`. That is,

```
F(0) = 0, F(1) = 1
F(n) = F(n - 1) + F(n - 2), for n > 1.
```

Given `n`, calculate `F(n)`.

 

**Example 1:**

```
Input: n = 2
Output: 1
Explanation: F(2) = F(1) + F(0) = 1 + 0 = 1.
```

```python
class Solution:
    def fib(self, n: int) -> int:
        if n <= 1:
            return n
        a, b, c = 0, 1, 0
        for i in range(n-1):
            c = a + b
            a, b = b, c
        return c
```

另外两种数学解法见https://leetcode-cn.com/problems/fibonacci-number/solution/fei-bo-na-qi-shu-by-leetcode-solution-o4ze/

#### 537. Complex Number Multiplication

A [complex number](https://en.wikipedia.org/wiki/Complex_number) can be represented as a string on the form `"**real**+**imaginary**i"` where:

- `real` is the real part and is an integer in the range `[-100, 100]`.
- `imaginary` is the imaginary part and is an integer in the range `[-100, 100]`.
- `i2 == -1`.

Given two complex numbers `num1` and `num2` as strings, return *a string of the complex number that represents their multiplications*.

 

**Example 1:**

```
Input: num1 = "1+1i", num2 = "1+1i"
Output: "0+2i"
Explanation: (1 + i) * (1 + i) = 1 + i2 + 2 * i = 2i, and you need convert it to the form of 0+2i.
```

```python
class Solution(object):
    def complexNumberMultiply(self, a, b):
        a_t, a_f = int(a.split('+')[0]), int(a.split('+')[1].split('i')[0])
        b_t, b_f = int(b.split('+')[0]), int(b.split('+')[1].split('i')[0])
        res_t, res_f = a_t * b_t - a_f * b_f, a_t * b_f + a_f * b_t
        return str(res_t) + "+" + str(res_f) + "i"
```



#### 541. Reverse String II

Given a string `s` and an integer `k`, reverse the first `k` characters for every `2k` characters counting from the start of the string.

If there are fewer than `k` characters left, reverse all of them. If there are less than `2k` but greater than or equal to `k` characters, then reverse the first `k` characters and left the other as original.

 

**Example 1:**

```
Input: s = "abcdefg", k = 2
Output: "bacdfeg"
```

```python
class Solution(object):
    def reverseStr(self, s, k):
        """
        :type s: str
        :type k: int
        :rtype: str
        """
        from functools import reduce
        # turn s into a list 
        s = list(s)
        
        # another way to simply use a[::-1], but i feel this is easier to understand
        def reverse(s):
            left, right = 0, len(s) - 1
            while left < right:
                s[left], s[right] = s[right], s[left]
                left += 1
                right -= 1
            return s
        
        # make sure we reverse each 2k elements 
        for i in range(0, len(s), 2*k):
            s[i:(i+k)] = reverse(s[i:(i+k)])
        
        # combine list into str.
        return reduce(lambda a, b: a+b, s) # same as "".join(s)
```





#### 738. Monotone Increasing Digits

An integer has **monotone increasing digits** if and only if each pair of adjacent digits `x` and `y` satisfy `x <= y`.

Given an integer `n`, return *the largest number that is less than or equal to* `n` *with **monotone increasing digits***.

 

**Example 1:**

```
Input: n = 10
Output: 9
```

```python
class Solution:
    def monotoneIncreasingDigits(self, n: int) -> int:
        a = list(str(n))
        for i in range(len(a)-1,0,-1):
            if int(a[i]) < int(a[i-1]):
                a[i-1] = str(int(a[i-1]) - 1)
                a[i:] = '9' * (len(a) - i)  #python不需要设置flag值，直接按长度给9就好了
        return int("".join(a)) 

```



#### 746. Min Cost Climbing Stairs

You are given an integer array `cost` where `cost[i]` is the cost of `ith` step on a staircase. Once you pay the cost, you can either climb one or two steps.

You can either start from the step with index `0`, or the step with index `1`.

Return *the minimum cost to reach the top of the floor*.

 

**Example 1:**

```
Input: cost = [10,15,20]
Output: 15
Explanation: Cheapest is: start on cost[1], pay that cost, and go to the top.
```

**Example 2:**

```
Input: cost = [1,100,1,1,1,100,1,1,100,1]
Output: 6
Explanation: Cheapest is: start on cost[0], and only step on 1s, skipping cost[3].
```

```python
class Solution:
    def minCostClimbingStairs(self, cost: List[int]) -> int:
        n = len(cost)
        dp = [0] * (n + 1)
        for i in range(2, n + 1):
            dp[i] = min(dp[i - 1] + cost[i - 1], dp[i - 2] + cost[i - 2])
        return dp[n]
```

```python
class Solution:
    def minCostClimbingStairs(self, cost: List[int]) -> int:
        n = len(cost)
        prev, curr = 0, 0
        for i in range(2, n + 1):
            nxt = min(curr + cost[i - 1], prev + cost[i - 2])
            prev, curr = curr, nxt
        return curr
```



#### 763. Partition Labels

You are given a string `s`. We want to partition the string into as many parts as possible so that each letter appears in at most one part.

Return *a list of integers representing the size of these parts*.

 

**Example 1:**

```
Input: s = "ababcbacadefegdehijhklij"
Output: [9,7,8]
Explanation:
The partition is "ababcbaca", "defegde", "hijhklij".
This is a partition so that each letter appears in at most one part.
A partition like "ababcbacadefegde", "hijhklij" is incorrect, because it splits s into less parts.
```

```python
class Solution:
    def partitionLabels(self, s: str) -> List[int]:
        lastPos = defaultdict(int)
        for i in range(len(s)):
            lastPos[s[i]] = i
            
        result = []
        left = 0
        right = 0
        for i in range(len(s)):
            right = max(right, lastPos[s[i]])
            if i == right:
                result.append(right - left + 1)
                left = right + 1
        return result
```



#### 844. Backspace String Compare

Given two strings `s` and `t`, return `true` *if they are equal when both are typed into empty text editors*. `'#'` means a backspace character.

Note that after backspacing an empty text, the text will continue empty.

 

**Example 1:**

```
Input: s = "ab#c", t = "ad#c"
Output: true
Explanation: Both s and t become "ac".
```

```python
class Solution(object):
    def backspaceCompare(self, S, T):
        def build(S):
            ans = []
            for c in S:
                if c != '#':
                    ans.append(c)
                elif ans:
                    ans.pop()
            return "".join(ans)
        return build(S) == build(T)
```

```python
class Solution(object):
    def backspaceCompare(self, S, T):
        def F(S):
            skip = 0
            for x in reversed(S):
                if x == '#':
                    skip += 1
                elif skip:
                    skip -= 1
                else:
                    yield x
        return all(x == y for x, y in itertools.zip_longest(F(S), F(T)))
```





#### 860. Lemonade Change

At a lemonade stand, each lemonade costs `$5`. 

Customers are standing in a queue to buy from you, and order one at a time (in the order specified by `bills`).

Each customer will only buy one lemonade and pay with either a `$5`, `$10`, or `$20` bill. You must provide the correct change to each customer, so that the net transaction is that the customer pays $5.

Note that you don't have any change in hand at first.

Return `true` if and only if you can provide every customer with correct change.

 

**Example 1:**

```
Input: [5,5,5,10,20]
Output: true
Explanation: 
From the first 3 customers, we collect three $5 bills in order.
From the fourth customer, we collect a $10 bill and give back a $5.
From the fifth customer, we give a $10 bill and a $5 bill.
Since all customers got correct change, we output true.
```

```python
class Solution(object):
    def lemonadeChange(self, bills):
        five = ten = 0
        for bill in bills:
            if bill == 5:
                five += 1
            elif bill == 10:
                if not five: return False
                five -= 1
                ten += 1
            else:
                if ten and five:
                    ten -= 1
                    five -= 1
                elif five >= 3:
                    five -= 3
                else:
                    return False
        return True
```





#### 977. Squares of a Sorted Array

Given an integer array `nums` sorted in **non-decreasing** order, return *an array of **the squares of each number** sorted in non-decreasing order*.

 

**Example 1:**

```
Input: nums = [-4,-1,0,3,10]
Output: [0,1,9,16,100]
Explanation: After squaring, the array becomes [16,1,0,9,100].
After sorting, it becomes [0,1,9,16,100].
```

```python
def sortedSquares(self, A):
    answer = collections.deque()
    l, r = 0, len(A) - 1
    while l <= r:
        left, right = abs(A[l]), abs(A[r])
        if left > right:
            answer.appendleft(left * left)
            l += 1
        else:
            answer.appendleft(right * right)
            r -= 1
    return list(answer)
```



#### 1005. Maximize Sum Of Array After K Negations

Given an array `nums` of integers, we **must** modify the array in the following way: we choose an `i` and replace `nums[i]` with `-nums[i]`, and we repeat this process `k` times in total. (We may choose the same index `i` multiple times.)

Return the largest possible sum of the array after modifying it in this way.

 

**Example 1:**

```
Input: nums = [4,2,3], k = 1
Output: 5
Explanation: Choose indices (1,) and nums becomes [4,-2,3].
```

```python
class Solution:
    def largestSumAfterKNegations(self, A: List[int], K: int) -> int:
        A = sorted(A, key=abs) # 将A按绝对值从小到大排列
        for i in range(len(A)-1,-1,-1):
            if K > 0 and A[i] < 0:
                A[i] *= -1
                K -= 1
        if K > 0:
            A[0] *= (-1)**K #取A最后一个数只需要写-1
        return sum(A)
```



#### 1047. Remove All Adjacent Duplicates In String

You are given a string `s` consisting of lowercase English letters. A **duplicate removal** consists of choosing two **adjacent** and **equal** letters and removing them.

We repeatedly make **duplicate removals** on `s` until we no longer can.

Return *the final string after all such duplicate removals have been made*. It can be proven that the answer is **unique**.

 

**Example 1:**

```
Input: s = "abbaca"
Output: "ca"
Explanation: 
For example, in "abbaca" we could remove "bb" since the letters are adjacent and equal, and this is the only possible move.  The result of this move is that the string is "aaca", of which only "aa" is possible, so the final string is "ca".
```

```python
class Solution:
    def removeDuplicates(self, s: str) -> str:
        t = list()
        for i in s:
            if t and t[-1] == i:
                t.pop(-1)
            else:
                t.append(i)
        return "".join(t)  # 字符串拼接
```

#### 1086. High Five

dict and sort



#### 1306. Jump Game III

Given an array of non-negative integers `arr`, you are initially positioned at `start` index of the array. When you are at index `i`, you can jump to `i + arr[i]` or `i - arr[i]`, check if you can reach to **any** index with value 0.

Notice that you can not jump outside of the array at any time.

**Example 1:**

```
Input: arr = [4,2,3,0,3,1,2], start = 5
Output: true
Explanation: 
All possible ways to reach at index 3 with value 0 are: 
index 5 -> index 4 -> index 1 -> index 3 
index 5 -> index 6 -> index 4 -> index 1 -> index 3 
```

```python
class Solution:
    def canReach(self, arr: List[int], start: int) -> bool:
        if arr[start] == 0:
            return True
        
        n = len(arr)
        used = set()
        q = collections.deque([start])
        
        while len(q) > 0:
            u = q.popleft()
            for v in [u + arr[u], u - arr[u]]:
                if 0 <= v < n and v not in used:
                    if arr[v] == 0:
                        return True
                    q.append(v)
                    used.add(v)
                    
        return False
```





#### 1888. Minimum Number of Flips to Make the Binary String Alternating

You are given a binary string `s`. You are allowed to perform two types of operations on the string in any sequence:

- **Type-1: Remove** the character at the start of the string `s` and **append** it to the end of the string.
- **Type-2: Pick** any character in `s` and **flip** its value, i.e., if its value is `'0'` it becomes `'1'` and vice-versa.

Return *the **minimum** number of **type-2** operations you need to perform* *such that* `s` *becomes **alternating**.*

The string is called **alternating** if no two adjacent characters are equal.

- For example, the strings `"010"` and `"1010"` are alternating, while the string `"0100"` is not.

 

**Example 1:**

```
Input: s = "111000"
Output: 2
Explanation: Use the first operation two times to make s = "100011".
Then, use the second operation on the third and sixth elements to make s = "101010".
```

```python
# sliding window
class Solution:
    def minFlips(self, s: str) -> int:
        n = len(s)
        s = s + s
        alt1, alt2 = "", ""
        for i in range(len(s)):
            alt1 += "0" if i % 2 else "1"
            alt2 += "1" if i % 2 else "0"
            
        res = len(s)
        diff1, diff2 = 0, 0
        l = 0
        for r in range(len(s)):
            if s[r] != alt1[r]:
                diff1 += 1
            if s[r] != alt2[r]:
                diff2 += 1
                
            if (r - l + 1) > n:
                if s[l] != alt1[l]:
                    diff1 -= 1
                if s[l] != alt2[l]:
                    diff2 -= 1
                l += 1
                
            if (r - l + 1) == n:
                res = min(res, diff1, diff2)
                
        return res
```





#### 1897. Redistribute Characters to Make All Strings Equal

You are given an array of strings `words` (**0-indexed**).

In one operation, pick two **distinct** indices `i` and `j`, where `words[i]` is a non-empty string, and move **any** character from `words[i]` to **any** position in `words[j]`.

Return `true` *if you can make **every** string in* `words` ***equal** using **any** number of operations*, *and* `false` *otherwise*.

 

**Example 1:**

```
Input: words = ["abc","aabc","bc"]
Output: true
Explanation: Move the first 'a' in words[1] to the front of words[2],
to make words[1] = "abc" and words[2] = "abc".
All the strings are now equal to "abc", so return true.
```

```python
class Solution:
    def makeEqual(self, words: List[str]) -> bool:
        freq = defaultdict(int)
        for word in words: 
            for ch in word: 
                freq[ch] += 1

        return all(x % len(words) == 0 for x in freq.values())
```





#### 1898. Maximum Number of Removable Characters

You are given two strings `s` and `p` where `p` is a **subsequence** of `s`. You are also given a **distinct 0-indexed** integer array `removable` containing a subset of indices of `s` (`s` is also **0-indexed**).

You want to choose an integer `k` (`0 <= k <= removable.length`) such that, after removing `k` characters from `s` using the **first** `k` indices in `removable`, `p` is still a **subsequence** of `s`. More formally, you will mark the character at `s[removable[i]]` for each `0 <= i < k`, then remove all marked characters and check if `p` is still a subsequence.

Return *the **maximum*** `k` *you can choose such that* `p` *is still a **subsequence** of* `s` *after the removals*.

A **subsequence** of a string is a new string generated from the original string with some characters (can be none) deleted without changing the relative order of the remaining characters.

 

**Example 1:**

```
Input: s = "abcacb", p = "ab", removable = [3,1,0]
Output: 2
Explanation: After removing the characters at indices 3 and 1, "abcacb" becomes "accb".
"ab" is a subsequence of "accb".
If we remove the characters at indices 3, 1, and 0, "abcacb" becomes "ccb", and "ab" is no longer a subsequence.
Hence, the maximum k is 2.
```

```python
class Solution:
    def maximumRemovals(self, s: str, p: str, removable: List[int]) -> int:

        def subSeq(m):
            i = j = 0
            remove = set(removable[:m+1])
            while i < len(s) and j < len(p):
                if i in remove:
                    i += 1
                    continue
                if s[i] == p[j]:
                    i += 1
                    j += 1
                else:
                    i += 1
            return j == len(p)

        left, right = 0, len(removable) - 1
        while left <= right:
            mid = left + (right - left) // 2
            if subSeq(mid):
                left = mid + 1
            else:
                right = mid - 1
        return left
```



#### 1899. Merge Triplets to Form Target Triplet

A **triplet** is an array of three integers. You are given a 2D integer array `triplets`, where `triplets[i] = [ai, bi, ci]` describes the `ith` **triplet**. You are also given an integer array `target = [x, y, z]` that describes the **triplet** you want to obtain.

Return `true` *if it is possible to obtain the* `target` ***triplet*** `[x, y, z]` *as an **element** of* `triplets`*, or* `false` *otherwise*.

**Example 1:**

```
Input: triplets = [[2,5,3],[1,8,4],[1,7,5]], target = [2,7,5]
Output: true
Explanation: Perform the following operations:
- Choose the first and last triplets [[2,5,3],[1,8,4],[1,7,5]]. Update the last triplet to be [max(2,1), max(5,7), max(3,5)] = [2,7,5]. triplets = [[2,5,3],[1,8,4],[2,7,5]]
The target triplet [2,7,5] is now an element of triplets.
```

```python
class Solution:
    def mergeTriplets(self, triplets: List[List[int]], target: List[int]) -> bool:
        ac=[0,0,0]
        if ac==target:
            return True
        for i in triplets:
            if i==target:
                return True
            if target[0]>=i[0] and target[1]>=i[1] and target[2]>=i[2]:
                ac=[max(ac[0],i[0]),max(ac[1],i[1]),max(ac[2],i[2])]
            if ac==target:
                return True
        return False
```



### SQL

#### 180. Consecutive Numbers

```mysql
with t as 
(
    select 
        Num, Id + 1 - row_number() over(partition by Num) diff
    from Logs
)

select distinct Num as ConsecutiveNums
from
    t
group by Num, diff
having count(*) >= 3
```



#### 185. Department Top Three Salaries

```mysql
select 
    Department, Employee, Salary
from 
(
    select
        a.Name as Employee,
        b.Name as Department,
        a.Salary,
        dense_rank() over(partition by DepartmentId order by Salary desc) as rk
    from 
        Employee a join Department b on a.DepartmentId = b.Id
) t
where rk <= 3
```



#### 511

```mysql
select player_id, min(device_id) first_login 
from Activity
group by player_id
```

#### 512

```mysql
select player_id, device_id 
from Activity
where (player_id, event_date) in 
(
    select player_id, min(event_date) event_date from Activity group by player_id
)
```

```mysql
select
    player_id, device_id
from
(
    select 
        player_id, device_id,
        row_number() over(partition by player_id order by event_date) as rn
    from 
        Activity
) t
where 
    rn = 1
```

#### 534. Game Play Analysis III

```mysql
# 内联
select t1.player_id, t1.event_date, sum(t2.games_played) as games_played_so_far
from Activity t1, Activity t2
where t1.player_id = t2.player_id
and t1.event_date >= t2.event_date
group by t1.player_id, t1.event_date
```

```mysql
select 
    player_id, event_date, 
    sum(games_played) over(partition by player_id order by event_date) as games_played_so_far
from 
    Activity
```



#### 550

```mysql
with t as 
(
    select 
        player_id,
        lead(event_date, 1) over(partition by player_id order by event_date) -
        min(event_date) over(partition by player_id order by event_date) diff
    from 
        Activity
)

select round((select count(*) from t where diff = 1) /
             (select count(distinct player_id) from Activity), 2) fraction
```

```mysql
# 外联
select round(avg(t1.event_date is not null), 2) as fraction
from Activity t1 right join 
(
    select player_id, min(event_date) as login
    from Activity
    group by player_id
) t2
on t1.player_id = t2.player_id 
and datediff(t1.event_date, login) = 1
```

#### 569. Median Employee Salary

```mysql
with t as(
    select 
        Id, Company, Salary,
        row_number() over(partition by Company order by salary) as rk,
        count(salary) over(partition by company) as cnt
    from 
        Employee
)

select
    Id, Company, Salary
from
    t
where
    rk between cnt/2 and cnt/2+1
order by
    company, salary
```



#### 570

```mysql
select Name
from Employee t1 join (
    select ManagerId 
    from Employee
    group by ManagerId
    having count(ManagerId) >= 5
) t2
on t1.Id = t2.ManagerId
```

#### 571

```mysql
select avg(number) median
from
    (select number,
        sum(frequency) over(order by number) asc_accum,
        sum(frequency) over(order by number desc) desc_accum
        from numbers) t1,
    (select sum(frequency) as total from numbers) t2
where t1.asc_accum >= total/2 and t1.desc_accum >= total/2
```

#### 574. Winning Candidate M

```mysql
select Name
from (
  select CandidateId as id
  from Vote
  group by CandidateId
  order by count(id) desc
  limit 1
) as Winner join Candidate
on Winner.id = Candidate.id
```

#### 577. Employee Bonus

```mysql
select name, bonus
from Employee t1 left join Bonus t2
using(empId)
where bonus is NULL or bonus < 1000
```

#### 578. Get Highest Answer Rate Question

```mysql
select t1.question_id as survey_log from
(
    select question_id, count(*) as total
    from survey_log
    where action = 'show'
    group by question_id
) t1
join
(
    select question_id, count(*) as answered
    from survey_log
    where action = 'answer'
    group by question_id
) t2
on t1.question_id = t2.question_id
order by answered/total desc
limit 1
```

```mysql
select question_id as survey_log
from survey_log
group by question_id
order by sum(if(action = 'answer', 1, 0)) / sum(if(action = 'show', 1, 0)) desc
limit 1
```

#### 580. Count Student Number in Departments

```mysql
select dept_name, sum(if(student_id is not null, 1, 0)) as student_number from
# COUNT(student_id) AS student_number
department left join student using(dept_id)
group by department.dept_id
order by student_number desc, dept_name
```

#### 584. Find Customer Referee

```mysql
select name 
from customer 
where referee_id != 2 or referee_id is NULL
```

#### 585. Investments in 2016

```mysql
select sum(tiv_2016) as TIV_2016 from insurance 
where tiv_2015 in 
(
    select tiv_2015 from insurance
    group by tiv_2015
    having count(*) > 1
)  
and concat(LAT,LON) in
(
    select concat(LAT,LON) from insurance
    group by LAT, LON
    having count(*) = 1
)
```

```mysql
select round(sum(tiv_2016), 2) as TIV_2016 from
(
    select *,
    count(*) over(partition by TIV_2015) as c1,
    count(*) over(partition by LAT, LON) as c2
    from insurance
) a
where a.c1 > 1 and a.c2 < 2
```



#### 586

```mysql
select customer_number
from Orders 
group by customer_number
order by count(*) desc
limit 1
```

#### 597. Friend Requests I: Overall Acceptance Rate

```mysql
with t1 as
(
    select count(distinct requester_id, accepter_id) as accept
    from RequestAccepted
),

t2 as
(
    select count(distinct sender_id, send_to_id) as send
    from FriendRequest
)

select round(
    ifnull((select accept from t1)/
    (select send from t2), 0), 2
)  accept_rate
```

#### 602. Friend Requests II: Who Has the Most Friends

```mysql
with t1 as
(
    select requester_id as id, count(*) as num
    from request_accepted
    group by requester_id 
),
t2 as 
(
    select accepter_id as id, count(*) as num
    from request_accepted
    group by accepter_id 
),
t3 as(
    select t1.id, (ifnull(t1.num, 0) + ifnull(t2.num, 0)) as num
    from t1 left join t2 using(id)
)

select id, num
from t3 
order by num desc
limit 1
```



#### 603. Consecutive Available Seats

```mysql
select distinct a.seat_id
from cinema a join cinema b 
on 
    abs(a.seat_id - b.seat_id) = 1
    and a.free = true and b.free = true
order by 
    a.seat_id
```

#### 607

```mysql
select s.name
from 
    salesperson s 
    left join orders o on s.sales_id = o.sales_id
    left join company c on c.com_id = o.com_id
group by
    s.name
having
    SUM(IF(c.name = 'RED', 1, 0))  = 0
order by
    s.sales_id
```

```python
select s.name
from salesperson s
where
s.sales_id in (
    select o.sales_id
    from orders o left join company c
    on c.com_id = o.com_id
    where c.name not in ('RED')
)
```

#### 608. Tree Node

```mysql
select distinct t1.id,
    case
        when t1.p_id is null then 'Root' 
        when t2.id is null then 'Leaf'
        else 'Inner'
    end Type
from tree t1 left join tree t2 on t1.id = t2.p_id
order by t1.id
```



#### 610

```mysql
select x, y, z, 
    case
        when x + y > z and x + z > y and z + y > x then 'Yes'
        else 'No'
    end as triangle
from triangle;
```

```mysql
select x, y, z, 
    if(x + y > z and x + z > y and z + y > x, 'Yes', 'No') as triangle
from triangle;
```

#### 612. Shortest Distance in a Plane

```mysql
select round(sqrt(min(power(a.x - b.x, 2) + power(a.y - b.y, 2))),2) as shortest 
from point_2d a join point_2d b
where a.x < b.x or a.y < b.y
```



#### 613

```mysql
select min(abs(t1.x - t2.x)) as shortest
from point t1, point t2
where t1.x != t2.x
```

#### 614. Second Degree Follower

```mysql
select t1.follower, count(distinct t2.follower) as num
from follow t1 join follow t2 on t1.follower = t2.followee
group by t1.follower
```

#### 619. Biggest Single Number

```mysql
with t1 as
(
    select num, count(*) as freq
    from my_numbers
    group by num
    having freq = 1
)

select max(num) as num from t1
```

#### 1045. Customers Who Bought All Products

```mysql
select customer_id
from Customer
group by customer_id
having count(distinct product_key) in 
(select count(distinct product_key) from Product)
```

#### 1050. Actors and Directors Who Cooperated At Least Three Times

```mysql
select actor_id, director_id
from ActorDirector
group by actor_id, director_id
having count(concat(actor_id, director_id)) >= 3
```

#### 1164. Product Price at a Given Date

```mysql
with t2 as(
    select product_id, new_price,
        row_number() over(partition by product_id order by change_date desc) rn
    from Products
    where change_date <= '2019-08-16'
),

t3 as(
    select product_id, new_price
    from t2
    where rn = 1
)

select distinct t1.product_id, ifnull(t3.new_price, 10) as price
from products t1 left join t3 on t1.product_id = t3.product_id
```

```mysql
select p1.product_id, ifnull(p2.new_price, 10) as price
from (
    select distinct product_id
    from products
) as p1 -- 所有的产品
left join (
    select product_id, new_price 
    from products
    where (product_id, change_date) in (
        select product_id, max(change_date)
        from products
        where change_date <= '2019-08-16'
        group by product_id
    )
) as p2 -- 在 2019-08-16 之前有过修改的产品和最新的价格
on p1.product_id = p2.product_id
```



#### 1070. Product Sales Analysis III

```mysql
with t1 as(
    select product_id, year, quantity, price,
    rank() over(partition by product_id order by year) as rk
    from Sales
)

select product_id, year first_year, quantity, price
from t1
where rk=1
```

#### 1173 Immediate Food Delivery I

```mysql
select round (
    sum(order_date = customer_pref_delivery_date) /
    count(*) * 100,
    2
) as immediate_percentage
from Delivery
```

#### 1174. Immediate Food Delivery II

```mysql
select round(sum(if(order_date = customer_pref_delivery_date, 1, 0))/
     count(*) * 100, 2) immediate_percentage
from Delivery
where (customer_id, order_date) in (
    select customer_id, min(order_date)
    from Delivery
    group by customer_id
)
```



#### 1075. Project Employees I

```mysql
select project_id, round(avg(experience_years),2) as average_years
from Project left join Employee using(employee_id)
group by project_id
```

#### 1076. Project Employees II

```mysql
select project_id
from project
group by project_id
having count(*) >=all 
(select count(*) amount
from project 
group by project_id)
```

```mysql
with t1 as (
    select 
        project_id,
        rank() over(order by count(*) desc) rk
    from project
    group by project_id
)

select project_id from t1 where rk = 1
```

#### 1077. Project Employees III

```mysql
with t1 as(
    select 
        project_id, employee_id, 
        rank() over(partition by project_id order by experience_years desc) as rk
    from project join employee using(employee_id)
)

select project_id, employee_id
from t1
where rk = 1
```

#### 1082. Sales Analysis I

```mysql
select seller_id
from sales
group by seller_id
having sum(price) >= all(
    select sum(price)
    from sales
    group by seller_id
)
```

```mysql
with t1 as(
    select 
        seller_id, 
        rank() over(order by sum(price) desc) as rk
    from sales
    group by seller_id
)

select seller_id from t1 where rk = 1
```

#### 1083. Sales Analysis II

```mysql
with t1 as(
    select buyer_id,
    case when product_name = 'S8' then 1 else 0 end s8,
    case when product_name = 'iPhone' then 1 else 0 end ip
    from sales s left join product p using(product_id)
)

select buyer_id
from t1
group by buyer_id
having sum(s8) > 0 and sum(ip) = 0
```

```mysql
select buyer_id   
from Sales left join Product using(product_id)
group by buyer_id 
having sum(product_name='S8')>0 and sum(product_name='iPhone')=0
```

```mysql
select distinct buyer_id
from Sales
where buyer_id not in (
    select buyer_id
    from Product p,Sales s
    where p.product_id = s.product_id
    and p.product_name = "iPhone"
) 
and buyer_id in (
    select buyer_id
    from Product p,Sales s
    where p.product_id = s.product_id
    and p.product_name = "S8"
)
```

#### 1084. Sales Analysis III

```mysql
select product_id, product_name
from sales left join product using(product_id)
group by product_id
having max(sale_date) <= '2019-03-31' and min(sale_date) >= '2019-01-01'
```

```mysql
select p.product_id, p.product_name
from Product p, Sales s
where p.product_id = s.product_id
group by s.product_id
having(sum(sale_date between '2019-01-01' and '2019.03-31') = count(*))
```



#### 1098. Unpopular Books

```mysql
select t1.book_id, t1.name
from Books t1 left join Orders on t1.book_id = Orders.book_id 
and dispatch_date >= date('2018-06-23') # 这个条件不能在where后否则忽略null
where available_from < date('2019-05-23')
group by book_id
having ifnull(sum(quantity), 0) < 10
```

```mysql
with t1 as
(
    select *
    from books
    where available_from <= date('2019-05-23')
)

select t1.book_id, t1.name
from t1 left join orders t2 on t1.book_id = t2.book_id
and dispatch_date > date('2018-06-23')
group by t1.book_id
having ifnull(sum(quantity),0) < 10
```



#### 1107. New Users Daily Count

```mysql
with t as 
(
    select user_id, min(activity_date) as login_date
    from Traffic
    where activity = 'login'
    group by user_id
) 

select login_date, count(user_id) as user_count
from t
where datediff(@today := '2019-06-30', t.login_date) <= 90
group by login_date
```

#### 1112. Highest Grade For Each Student

```mysql
with t1 as(
    select 
        student_id, course_id, grade,
        row_number() over(partition by student_id order by grade desc, course_id) rn
    from 
        Enrollments
)

select student_id, course_id, grade
from t1 where rn = 1
```

#### 1113. Reported Posts

```mysql
select extra as report_reason, count(distinct post_id) as report_count
from Actions
where action = 'report' and action_date = '2019-07-04'
group by extra
```

#### 1126. Active Businesses

```mysql
with t1 as(
    select event_type, avg(occurences) as avg
    from Events
    group by event_type
)

select business_id
from Events join t1 using(event_type)
group by business_id
having sum(if(occurences > avg, 1, 0)) >= 2
```

#### 1132. Reported Posts II

```mysql
SELECT ROUND(AVG(proportion) * 100, 2) AS average_daily_percent  
FROM (
    SELECT actions.action_date, COUNT(DISTINCT removals.post_id)/COUNT(DISTINCT actions.post_id) AS proportion
    FROM actions
    LEFT JOIN removals
    ON actions.post_id = removals.post_id
    WHERE extra = 'spam' 
    GROUP BY actions.action_date
) a
```

```mysql
SELECT ROUND(AVG(IFNULL(remove.cnt, 0)/total.cnt) * 100, 2) AS average_daily_percent
FROM (
    SELECT action_date, COUNT(DISTINCT post_id) AS cnt
    FROM actions
    WHERE extra = 'spam'
    GROUP BY action_date
) total
LEFT JOIN (
    SELECT action_date, COUNT(DISTINCT post_id) AS cnt
    FROM actions
    WHERE extra = 'spam' AND post_id IN (SELECT post_id FROM Removals)
    GROUP BY action_date
) remove 
ON total.action_date = remove.action_date
```

#### 1142. User Activity for the Past 30 Days II

```mysql
select round(ifnull(count(distinct session_id)/count(distinct user_id), 0), 2) as average_sessions_per_user 
from Activity
where activity_date > '2019-06-27'
```

#### 1148. Article Views I

```mysql
select distinct author_id as id
from Views
where author_id = viewer_id
order by author_id
```

#### 1149. Article Views II

```mysql
select distinct viewer_id as id
from Views
group by view_date, viewer_id
having count(distinct article_id) > 1
order by viewer_id
```

#### 1158. Market Analysis I

```mysql
select Users.user_id as buyer_id, join_date, ifnull(UserBuy.cnt, 0) as orders_in_2019
from Users
left join (
    select buyer_id, count(order_id) cnt 
    from Orders
    where order_date between '2019-01-01' and '2019-12-31'
    group by buyer_id
) UserBuy
on Users.user_id = UserBuy.buyer_id
```



#### 1179

```mysql
SELECT id,
IF(`month`='Jan',revenue,NULL) Jan_Revenue,
IF(`month`='Feb',revenue,NULL) Feb_Revenue,
IF(`month`='Mar',revenue,NULL) Mar_Revenue,
IF(`month`='Apr',revenue,NULL) Apr_Revenue,
IF(`month`='May',revenue,NULL) May_Revenue,
IF(`month`='Jun',revenue,NULL) Jun_Revenue,
IF(`month`='Jul',revenue,NULL) Jul_Revenue,
IF(`month`='Aug',revenue,NULL) Aug_Revenue,
IF(`month`='Sep',revenue,NULL) Sep_Revenue,
IF(`month`='Oct',revenue,NULL) Oct_Revenue,
IF(`month`='Nov',revenue,NULL) Nov_Revenue,
IF(`month`='Dec',revenue,NULL) Dec_Revenue
FROM Department
GROUP BY id;
```

```mysql
SELECT id,
SUM(CASE `month` WHEN 'Jan' THEN revenue END) Jan_Revenue,
SUM(CASE `month` WHEN 'Feb' THEN revenue END) Feb_Revenue,
SUM(CASE `month` WHEN 'Mar' THEN revenue END) Mar_Revenue,
SUM(CASE `month` WHEN 'Apr' THEN revenue END) Apr_Revenue,
SUM(CASE `month` WHEN 'May' THEN revenue END) May_Revenue,
SUM(CASE `month` WHEN 'Jun' THEN revenue END) Jun_Revenue,
SUM(CASE `month` WHEN 'Jul' THEN revenue END) Jul_Revenue,
SUM(CASE `month` WHEN 'Aug' THEN revenue END) Aug_Revenue,
SUM(CASE `month` WHEN 'Sep' THEN revenue END) Sep_Revenue,
SUM(CASE `month` WHEN 'Oct' THEN revenue END) Oct_Revenue,
SUM(CASE `month` WHEN 'Nov' THEN revenue END) Nov_Revenue,
SUM(CASE `month` WHEN 'Dec' THEN revenue END) Dec_Revenue
FROM Department
GROUP BY id;
```

#### 1193. Monthly Transactions I

```mytsql
SELECT DATE_FORMAT(trans_date, '%Y-%m') AS month,
    country,
    COUNT(*) AS trans_count,
    sum(IF(state = 'approved', 1, 0)) AS approved_count,
    SUM(amount) AS trans_total_amount,
    SUM(IF(state = 'approved', amount, 0)) AS approved_total_amount
FROM Transactions
GROUP BY month, country
```

#### 1204. Last Person to Fit in the Bus

```mysql
select t1.person_name
from queue t1 left join queue t2 on t1.turn >= t2.turn
group by t1.person_id
having sum(t2.weight) <= 1000
order by t1.turn desc
limit 1 
```

```mysql
select t.person_name
from (
    select person_name, @pre := @pre + weight as weight
    from queue join (select @pre := 0) tmp
    order by turn
) t
where t.weight <= 1000
order by weight desc
limit 1
```



#### 1613. Find the Missing IDs

```mysql
with recursive t(n) as(
    select 1
    union all
    select 1 + n from t where n < (select max(customer_id) from Customers)
)

select n as ids
from Customers c right join t on c.customer_id = t.n
where c.customer_id is null
```

#### Weekly Contest

##### Weekly Contest 257

#### 1995. Count Special Quadruplets

BruteForce

```python
class Solution:
    def countQuadruplets(self, nums: List[int]) -> int:
        n = len(nums)
        count = 0
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    temp = nums[i] + nums[j] + nums[k]
                    count += nums[(k+1):].count(temp)
        return count
```



#### 1996. The Number of Weak Characters in the Game

two way sort

```python
class Solution:
    def numberOfWeakCharacters(self, properties: List[List[int]]) -> int:
        
        properties.sort(key = lambda x: (-x[0], x[1]))
        
        count = 0
        maxdf = 0
        
        for p in properties:
            if p[1] < maxdf:
                count += 1 
            maxdf = max(p[1], maxdf)
        return count
```

```python
class Solution:
    def numberOfWeakCharacters(self, properties: List[List[int]]) -> int:
        
        properties.sort(key=lambda x: (x[0], -x[1]))
        
        stack = []
        ans = 0
        
        for a, d in properties:
            while stack and stack[-1][0] < a and stack[-1][1] < d:
                stack.pop()
                ans += 1
            stack.append((a, d))
        return ans
```

#### 1997. First Day Where You Have Been in All the Rooms

DP

```python
class Solution:
    def firstDayBeenInAllRooms(self, nextVisit: List[int]) -> int:
        n = len(nextVisit)
        dp = [0] * n
        
        for i in range(1, n):
            dp[i] = (dp[i-1] + 1 + dp[i-1] - dp[nextVisit[i-1]] + 1)%1000000007
        
        return dp[-1]
```

#### 1998. GCD Sort of an Array

UnionFind

```python
class Solution:
    def gcdSort(self, nums: List[int]) -> bool:
        target = sorted(nums)
        a = set(nums)
        max_v = max(a)
        uf = UnionFind(max_v + 1)
        for i in range(2,max_v + 1):
            # 枚举i的倍数
            for j in range(i,max_v + 1,i):
                if j in a:
                    uf.union(j,i)
        
        for i in range(len(nums)):
            ux,uy = uf.find(nums[i]),uf.find(target[i])
            # 目标位置与当前位置未连通
            if ux != uy:
                return False
        return True


class UnionFind():
    def __init__(self, n):
        self.father = [i for i in range(n)]
        self.size = [1] * n
        self.count = n
        
    def find(self, x):
        while x != self.father[x]:
            self.father[x] = self.find(self.father[x]) # 压缩路径
            x = self.father[x]
        return self.father[x]

    def union(self, x, y):
        father_x, father_y = self.find(x), self.find(y)
        if father_x != father_y:
            if self.size[father_x] > self.size[father_y]:
                father_x, father_y = father_y, father_x
            self.father[father_x] = father_y
            self.size[father_y] += self.size[father_x]
            self.count -= 1
```



#### Basics

![img](https://pica.zhimg.com/80/v2-b92f60067804733195466a192e0b3603_720w.jpg?source=1940ef5c)

#### References

https://github.com/SharingSource/LogicStack-LeetCode/wiki

https://programmercarl.com/

https://github.com/youngyangyang04/leetcode-master

https://runestone.academy/runestone/books/published/pythonds/index.html

https://github.com/wisdompeak/LeetCode

https://github.com/grandyang/leetcode

https://books.halfrost.com/leetcode/
