# 搜索

## 图的存储

图的存储方式以及 dfs 示例。

### 邻接矩阵

```c++
#include <iostream>
#include <cstring>
#include <algorithm>

using namespace std;

const int N = 1010;
int n, m, a, b, c;
int w[N][N];
bool vis[N];

void dfs(int u)
{
    vis[u] = true;
    for (int v = 1; v <= n; v++)
    {
        if (w[u][v])
        {
            cout << u << v << w[u][v] << endl;
            if (!vis[v])
                dfs(v);
        }
    }
}

int main()
{
    cin >> n >> m;
    while (m--)
    {
        cin >> a >> b >> c;
        w[a][b] = c;
    }
    dfs(1)
    return 0;
}
```

### 邻接表

```c++
//无边权
#include <iostream>
#include <cstring>
#include <algorithm>
#include<vector>

using namespace std;

const int N = 1010;
int n, m, a, b;
vector<int> e[N];
bool vis[N];

void dfs(int u)
{
    vis[u] = true;
    for (auto ed : e[u])
    {
        int v = ed;
        cout << u << v;
        if (!vis[v])
            dfs(v);
    }
}

int main()
{
    cin >> n >> m;
    while(m --)
    {
        cin >> a >> b;
        e[a].push_back(b);
    }
    return 0;
}

//有边权
struct edge
{
    int v, w; //边的终点v和权重w
}
vector<edge> e[N];

int main()
{
    cin >> n >> m;
    while(m --)
    {
        cin >> a >> b >> c;
        e[a].push_back({b, c});
    }
    return 0;
}
```

### 边集数组

```c++
#include <iostream>
#include <cstring>
#include <algorithm>
using namespace std;

const int N = 1010, M = 1010;
int n, m, a, b, c;
struct edge
{
    int u, v, w;
} e[M];
bool vis[N];

void dfs(int u)
{
    vis[u] = true;
    for (int i = 1; i <= m; i++)
    {
        if (e[i].u == u)
        {
            int v = e[i].v, w = e[i].w;
            cout << u << v << w;
            if (!vis[v])
                dfs(v);
        }
    }
}

int main()
{
    cin >> n >> m;
    for (int i = 1; i <= m; i++)
    {
        cin >> a >> b >> c;
        e[i] = {a, b, c};
    }
    return 0;
}
```

### 链式邻接表

```c++
struct edge 
{
    int v, w;
};
vector<edge> e;
vector<int> h[N];
```

- struct edge 为边的结构体类型，v 为终点，w 为权重。
- vector e 是一个边的集合，用于存储图中的所有边。
- h [N]：邻接表，h [i] 存储结点 i 相关的所有出边再 e 中的索引。

```c++
void add(int a, int b, int c) {
    e.push_back({b, c});
    h[a].push_back(e.size() - 1);
}
```

- `e.push_back({b, c});`：在边集 e 中添加一条新的边，起点为 a，终点为 b，权重为 c。
- `h[a].push_back(e.size() - 1);`：记录这条边在 e 中的索引，并储存在 h [a] 中，表示结点 a 的一条储边。`e.size() - 1` 是边在 e 中的索引

```c++
void dfs(int u)
{
    vis[u] = true;
    for (int i = 0; i < h[u].size(); i++)
    {
        int j = h[u][i];
        int v = e[j].v, w = e[j].w;
        cout << u << v << w;
        if (!vis[v])
            dfs(v);
    }
}

int main() {
    cin >> n >> m;
    for (int i = 1; i <= m; i++) {
        cin >> a >> b >> c;
        add(a, b, c); // 添加边 a->b
        add(b, a, c); // 添加边 b->a (因为图是无向的)
    }
    dfs(1);  
    return 0;
}
```

假设图中有以下边：

- (1, 2, 5)：从节点 1 到节点 2 的权重为 5 的边。
- (2, 3, 3)：从节点 2 到节点 3 的权重为 3 的边。

当我们调用 add 函数后，e 和 h 的状态如下：

- e (边集)：
  - e [0]：{2, 5}（代表边 1-> 2，权重 5） 
  - e [1]：{1, 5}（代表边 2-> 1，权重 5，表示无向图的另一方向）
  - e [2]：{3, 3}（代表边 2-> 3，权重 3）
  - e [3]：{2, 3}（代表边 3-> 2，权重 3）
- h (邻接表)：
  - h [1]：[0]（h [1] 表示节点 1 的出边集合，出边为 e [0]）
  - h [2]：[1, 2]（h [2] 表示节点 2 的出边集合，出边为 e [1] 和 e [2]）
  - h [3]：[3]（h [3] 表示节点 3 的出边集合，出边为 e [3]）

### 链式前向星

```c++
struct edge
{
    int v, w, ne;
} e[M];
int idx, h[N];
```

- struct edge 定义了边的结构体，v 为边的终点，w 为权重，ne 为下条边的索引
- idx 为当前边的索引，每添加一条边后递增
- h [N] 存储每个结点的第一条出边索引

```c++
void add(int a, int b, int c) {
    e[idx] = {b, c, h[a]}; // 将边 (a -> b, 权重为 c) 添加到边集中
    h[a] = idx++; // 更新节点 a 的第一条出边
}

void dfs(int u)
{
    vis[u] = true;
    for(int i = h[u]; ~i; i = e[i].ne)
    {
        int v = e[i].v, w = e[i].w;
        cout << u << v << w;
        if(!vis[v])
            dfs(v);
    }
}

int main() {
    cin >> n >> m; // 输入节点数和边数
    memset(h, -1, sizeof h); // 初始化邻接表
    for (int i = 1; i <= m; i++) {
        cin >> a >> b >> c; // 输入边的端点和权重
        add(a, b, c); // 添加边
        add(b, a, c); // 如果是无向图，则添加反向边
    }
    dfs(1)
    return 0;
}
```