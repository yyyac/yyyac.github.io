# 基础算法

```c++
#include <iostream>
#include <algorithm>
#include <cstring>

using namespace std;

const int N = 100;
int a[N], b[N];

// 插入排序
void insertsort(int a[], int n)
{
    for (int i = 2; i <= n; i++)
    {
        a[0] = a[i];
        int j = i - 1;
        while (a[0] < a[j])
        {
            a[j + 1] = a[j];
            j--;
        }
        a[j + 1] = a[0];
    }
}

// 折半插入排序
void binarysort(int a[], int n)
{
    int i, j, low, high, mid;
    for (int i = 2; i <= n; i++)
    {
        a[0] = a[i];
        low = 1, high = i - 1;
        while (low <= high)
        {
            mid = (low + high) / 2;
            if (a[mid] > a[0])
                high = mid - 1;
            else
                low = mid + 1;
        }
        for (int j = i - 1; j >= low; j--)
            a[j + 1] = a[0];
        a[low] = a[0];
    }
}

// 希尔排序
void shellsort(int a[], int n)
{
    int d, i, j;
    for (d = n / 2; d >= 1; d = d / 2)
    {
        for (i = d + 1; i <= n; i++)
        {
            if (a[i] < a[i - d])
            {
                a[0] = a[i];
                for (j = i - d; j > 0 && a[0] < a[j]; j--)
                    a[j + d] = a[j];
                a[j + d] = a[0];
            }
        }
    }
}

// 冒泡排序
void bubblesort(int a[], int n)
{
    for (int i = 0; i < n - 1; i++)
    {
        bool flag = false;
        for (int j = n - 1; j > i; j--)
        {
            if (a[j - 1] > a[j])
            {
                swap(a[j - 1], a[j]);
                flag = true;
            }
        }
        if (flag == false)
            return;
    }
}

// 快速排序
int Partition(int a[], int low, int high)
{
    int pivot = a[low];
    while (low < high)
    {
        while (low < high && a[high] >= pivot)
            high--;
        a[low] = a[high];
        while (low < high && a[low] <= pivot)
            low++;
        a[high] = a[low];
    }
    a[low] = pivot;
    return low;
}

void quicksort(int a[], int s, int t)
{
    if (s < t)
    {
        int pivotloc = Partition(a, s, t);
        quicksort(a, s, pivotloc - 1);
        quicksort(a, pivotloc + 1, t);
    }
}

// 简单选择排序
void slectsort(int a[], int n)
{
    for (int i = 0; i < n - 1; i++)
    {
        int j = i;
        for (int k = j + 1; k <= n; k++)
        {
            if (a[k] < a[j])
                j = k;
        }
        if (i != j)
            swap(a[i], a[j]);
    }
}

// 堆排序
void HeapAdjust(int a[], int k, int n)
{
    a[0] = a[k];
    for (int i = 2 * k; i <= n; i = i * 2)
    {
        if (i < n && a[i] < a[i + 1])
            i++;
        if (a[0] > a[i])
            break;
        else
        {
            a[k] = a[i];
            k = i;
        }
    }
    a[k] = a[0];
}

void BuildMaxHeap(int a[], int n)
{
    for (int i = n / 2; i > 0; i--)
        HeapAdjust(a, i, n);
}

void Heapsort(int a[], int n)
{
    BuildMaxHeap(a, n);
    for (int i = n; i > 1; i--)
    {
        swap(a[i], a[1]);
        HeapAdjust(a, 1, i - 1);
    }
}

// 归并排序

void merge(int a[], int low, int mid, int high)
{
    int i, j, k;
    for (int k = low; k <= high; k++)
        b[k] = a[k];

    for (i = low, j = mid + 1, k = i; i <= mid && j <= high; k++)
    {
        if (b[i] <= b[j])
            a[k] = b[i++];
        else
            a[k] = b[j++];
    }
    while (i <= mid)
        a[k++] = b[i++];
    while (j <= high)
        a[k++] = b[j++];
}

void mergesort(int a[], int low, int high)
{
    if (low < high)
    {
        int mid = (low + high) / 2;
        mergesort(a, low, mid);
        mergesort(a, mid + 1, high);
        merge(a, low, mid, high);
    }
}

```