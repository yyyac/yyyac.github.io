# 数组

## 二分查找

二分是在有序数组或者有序区间上查找的高效算法。可以采用红蓝边界法求解，基本思想为：

- 根据查找元素，将数组/区间分为两部分（红蓝区间）。
- 判断 mid 元素是否为蓝色，更新红蓝区间
- 根据题意最终返回红蓝边界

需要在查找前判断若 `a[0] > x || a[n - 1] < x`，则直接返回 `-1`；假设存在序列 `[1,4,6,7,8]`，查找元素为 `0`。初始 `l = -1, r = 5, mid = 2`，`q[mid]` 永远大于 x，故 `l` 不会更新，会发生数组越界，所以要提前处理。其中 L 永远指向蓝色区间，R 永远指向红色区间，所以初始化时 `l = -1, r = n`。

<figure markdown=span> ![](images/binary_search.jpg) </figure>

**算法模板**

```C++
int find(int a[], int n, int x)
{
    if(a[0] > x || a[n - 1] < x)
        return -1;
    int l = -1, r = n;  //
    while(l + 1 != r)
    {
        int mid = (l + r) / 2;
        if(IsBlue(mid)) l = mid;
        else r = mid;
    }
    return l or r;
}
```

### [二分查找](https://leetcode.cn/problems/binary-search/description/)

```C++
class Solution {
public:
    int search(vector<int>& nums, int target) {
        int n = nums.size() - 1;
        if(nums[0] > target || nums[n] < target)
            return -1;
        int l = -1, r = n + 1;
        while(l + 1 != r)
        {
            int mid = (l + r) / 2;
            if(nums[mid] <  target) l = mid;
            else r = mid;
        }
        if(nums[r] == target) return r;
        else return -1;
    }
};
```

### [搜索插入位置](https://leetcode.cn/problems/search-insert-position/description/)

```C++
class Solution {
public:
    int searchInsert(vector<int>& nums, int x) {
        int n = nums.size() - 1;
        if(nums[0] > x) return 0;
        if(nums[n] < x) return n + 1;
        int l = -1, r = n + 1;
        while(l + 1 != r)
        {
            int mid = (l + r) / 2;
            if(nums[mid] < x) l = mid;
            else r = mid;
        }
        return r;
    }
};
```