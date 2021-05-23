#### To do: Array 48

[toc]



### Binary Search

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
def searchRange(self, nums: List[int], target: int) -> List[int]:
        if len(nums) == 0:
            return [-1, -1]
        if nums[0] > target:
            return [-1, -1]
    
        start = 0
        end = len(nums) - 1
        
        start = self.find_first(nums, target, start, end)
        end = self.find_end(nums, target, start, end)
        
        return [start, end]
    
    def find_first(self, nums, target, start, end):
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
    
    def find_end(self, nums, target, start, end):
        while start + 1 < end:
            mid = start + (end - start) // 2
            if nums[mid] > target:
                end = mid
            else:
                start = mid
        if nums[end] == target:
            return end
        if nums[start] == target:
            return start
        return -1
        
```

#### 110. Balanced Binary Tree

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



### Two Pointer

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

#### 3. Longest Substring Without Repeating Characters

Given a string `s`, find the length of the **longest substring** without repeating characters.

**Example 1:**

```
Input: s = "abcabcbb"
Output: 3
Explanation: The answer is "abc", with the length of 3.
```

```python
def lengthOfLongestSubstring(self, s: str) -> int:
    start = -1
    max = 0
    d = {}

    for i in range(len(s)):
        if s[i] in d and d[s[i]] > start:
            start = d[s[i]]
            d[s[i]] = i
            else:
                d[s[i]] = i
                if i - start > max:
                    max = i - start
   return max
```







#### 9. Palindrome Number


```python
def isPalindrome(self, x: int) -> bool:
    x = str(x)
    left = 0
    right = len(x) - 1
    while left <= right:
        if abs(left - right) <=  1 and x[left] == x[right]:
            return True
        if x[left] == x[right]:
            left += 1
            right -= 1
        else:
            return False
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
    n = len(nums)
    for i in range(n):
        left, right = i + 1, n - 1
        while (left < right):
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
    count = 0
    for j in range(1, len(nums)):
        if nums[j] != nums[count]:
            count += 1
            nums[count] = nums[j]
    return count + 1
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
        """
        :type nums: List[int]
        :rtype: int
        """
        
        # Initialize the counter and the array index.
        i, count = 1, 1
        
        # Start from the second element of the array and process
        # elements one by one.
        while i < len(nums):
            
            # If the current element is a duplicate, 
            # increment the count.
            if nums[i] == nums[i - 1]:
                count += 1
                
                # If the count is more than 2, this is an
                # unwanted duplicate element and hence we 
                # remove it from the array.
                if count > 2:
                    nums.pop(i)
                    
                    # Note that we have to decrement the
                    # array index value to keep it consistent
                    # with the size of the array.
                    i-= 1
                
            else:
                
                # Reset the count since we encountered a different element
                # than the previous one
                count = 1
           
            # Move on to the next element in the array
            i += 1    
                
        return len(nums)
```

```python
class Solution(object):
    def removeDuplicates(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        
        # Initialize the counter and the second pointer.
        j, count = 1, 1
        
        # Start from the second element of the array and process
        # elements one by one.
        for i in range(1, len(nums)):
            
            # If the current element is a duplicate, 
            # increment the count.
            if nums[i] == nums[i - 1]:
                count += 1
            else:
                # Reset the count since we encountered a different element
                # than the previous one
                count = 1
            
            # For a count <= 2, we copy the element over thus
            # overwriting the element at index "j" in the array
            if count <= 2:
                nums[j] = nums[i]
                j += 1
                
        return j
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
    if n > m:
        nums1[:n] = nums2[:n]
```





### Recursive

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



#### 100. Same Tree

```python
def isSameTree(self, p: TreeNode, q: TreeNode) -> bool:
    if p is None and q is None:
        return True
    if p is not None and q is not None:
        return p.val == q.val and self.isSameTree(p.left, q.left) and self.isSameTree(p.right, q.right)
    return False
```

#### 101. Symmetric Tree

```python
def isSymmetric(self, root: TreeNode) -> bool:
    if root is None:
        return True
    return self.helper(root.left, root.right)

def helper(self, left, right):
    if left is None and right is None:
        return True
    if left is None or right is None or left.val != right.val:
        return False
    return self.helper(left.left, right.right) and self.helper(left.right, right.left)
```

#### 104. Maximum Depth of Binary Tree

```python
def maxDepth(self, root: TreeNode) -> int:
    if root is None:
        return 0 
    return max(self.maxDepth(root.left), self.maxDepth(root.right)) + 1
```

#### 111. Minimum Depth of Binary Tree

```python
def minDepth(self, root: TreeNode) -> int:
    if root is None:
        return 0
    if root.left and root.right:
        return min(self.minDepth(root.left), self.minDepth(root.right)) + 1
    else:
        return max(self.minDepth(root.left), self.minDepth(root.right)) + 1
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





### DP

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





### Backtracking

#### 17. Letter Combinations of a Phone Number

Algorithm: Greedy

```python
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

Algorithm: DFS

```python
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
        self.dfs(digits, dic, index+1, path+j, result)
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





#### 47. Permutations II

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
def permuteUnique(self, nums: List[int]) -> List[List[int]]:
    res = []
    used = [False] * len(nums)
    nums.sort()
    path = []
    self.dfs(nums, path, used, res)
    return res

def dfs(self, nums, path, used, res):
    if len(nums) == len(path):
        res.append(path)
        return
    for i in range(len(nums)):
        if used[i] or (i > 0 and nums[i] == nums[i-1] and not used[i-1]):
            continue
        used[i] = True
        self.dfs(nums, path + [nums[i]], used, res)
        used[i] = False
```

#### 79. Word Search

Algorithm: DFS

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



#### 55. Jump Game

Algorithm: Greedy. 

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
def maxProfit(self, prices: List[int]) -> int:
    max_profi, min_price = 0, float("inf")
    for i in prices:
        min_price = min(min_price, i)
        max_profi = max(max_profi, i - min_price)
    return max_profi
```



#### 122. Best Time to Buy and Sell Stock

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



#### 136. ! Single Number

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



#### References

https://books.halfrost.com/leetcode/
